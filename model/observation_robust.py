"""Composable M1--M4 modules for observation-robust landslide segmentation.

This is an independent implementation of the project's proposed mechanisms.
It is an implementation gate, not a released experiment or paper result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


SENSOR_ORDER = ("s1asc", "s1dsc", "s2")
SCENARIO_ORDER = (
    "none",
    "drop_modality",
    "drop_sar_orbit",
    "drop_contiguous_time",
    "drop_random_time",
)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise ValueError("cannot select a group-normalization divisor")


@dataclass(frozen=True)
class AblationSpec:
    """One preregistered cumulative M1--M4 ablation."""

    use_event_time: bool
    use_sensor_set_fusion: bool
    use_quality: bool
    use_structured_dropout: bool


ABLATION_SPECS = {
    "A0": AblationSpec(False, False, False, False),
    "A1": AblationSpec(True, False, False, False),
    "A2": AblationSpec(True, True, False, False),
    "A3": AblationSpec(True, True, True, False),
    "A4": AblationSpec(True, True, True, True),
}


@dataclass(frozen=True)
class StructuredDropoutResult:
    """Structured M4 mask and the realized per-sample scenario labels."""

    valid_time_mask: torch.Tensor
    scenarios: tuple[str, ...]


class EventRelativeTimeEncoder(nn.Module):
    """M1: confidence-attenuated mixture over one or more audited events."""

    def __init__(
        self,
        *,
        embedding_dim: int = 16,
        scales_days: Sequence[float] = (30.0, 90.0, 365.0),
        phase_window_days: float = 30.0,
        maximum_relative_days: float = 730.0,
    ) -> None:
        super().__init__()
        scales = tuple(float(value) for value in scales_days)
        if embedding_dim <= 0 or not scales or any(value <= 0 for value in scales):
            raise ValueError("invalid event-relative encoder dimensions/scales")
        if phase_window_days <= 0 or maximum_relative_days <= 0:
            raise ValueError("event-relative windows must be positive")
        self.embedding_dim = embedding_dim
        self.phase_window_days = float(phase_window_days)
        self.maximum_relative_days = float(maximum_relative_days)
        self.register_buffer("scales_days", torch.tensor(scales, dtype=torch.float32))
        raw_width = 1 + 2 * len(scales) + 3
        self.projection = nn.Sequential(
            nn.Linear(raw_width, embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )

    def forward(
        self,
        acquisition_days: torch.Tensor,
        valid_time_mask: torch.Tensor,
        event_days: torch.Tensor,
        event_confidence: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        if acquisition_days.ndim != 2:
            raise ValueError("acquisition days must have shape [B,T]")
        batch, steps = acquisition_days.shape
        if valid_time_mask.shape != (batch, steps):
            raise ValueError("valid-time mask must match acquisition days")
        if event_days.ndim != 2 or event_days.shape[0] != batch:
            raise ValueError("event days must have shape [B,E]")
        if event_confidence.shape != event_days.shape or event_mask.shape != event_days.shape:
            raise ValueError("event confidence/mask shapes must match event days")
        if not bool(event_mask.to(dtype=torch.bool).any(dim=1).all()):
            raise ValueError("M1 requires at least one audited event anchor per sample")
        if not bool(torch.isfinite(acquisition_days).all()):
            raise ValueError("acquisition days must be finite")
        event_mask = event_mask.to(device=event_days.device, dtype=torch.bool)
        if not bool(torch.isfinite(event_days[event_mask]).all()):
            raise ValueError("audited event days must be finite")
        if not bool(torch.isfinite(event_confidence[event_mask]).all()):
            raise ValueError("event confidence must be finite")
        if bool(
            (
                (event_confidence[event_mask] < 0)
                | (event_confidence[event_mask] > 1)
            ).any()
        ):
            raise ValueError("event confidence must lie in [0,1]")

        dtype = acquisition_days.dtype
        device = acquisition_days.device
        event_days = event_days.to(device=device, dtype=dtype)
        confidence = event_confidence.to(device=device, dtype=dtype)
        event_mask = event_mask.to(device=device)
        event_days = torch.where(event_mask, event_days, torch.zeros_like(event_days))
        confidence = torch.where(event_mask, confidence, torch.zeros_like(confidence))
        relative = acquisition_days[:, :, None] - event_days[:, None, :]
        relative = relative.clamp(
            min=-self.maximum_relative_days,
            max=self.maximum_relative_days,
        )
        normalized = relative / self.maximum_relative_days
        scales = self.scales_days.to(device=device, dtype=dtype)
        angles = relative[..., None] * (2.0 * math.pi / scales)
        phase_window = torch.as_tensor(
            self.phase_window_days, device=device, dtype=dtype
        )
        pre = torch.sigmoid(-relative / phase_window)
        near = torch.exp(-relative.abs() / phase_window)
        post = torch.sigmoid(relative / phase_window)
        raw_per_event = torch.cat(
            (
                normalized[..., None],
                torch.sin(angles),
                torch.cos(angles),
                pre[..., None],
                near[..., None],
                post[..., None],
            ),
            dim=-1,
        )

        masked_confidence = confidence * event_mask.to(dtype)
        confidence_sum = masked_confidence.sum(dim=1, keepdim=True)
        event_count = event_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype)
        fallback_weights = event_mask.to(dtype) / event_count
        normalized_weights = masked_confidence / confidence_sum.clamp_min(1e-12)
        weights = torch.where(
            confidence_sum > 0,
            normalized_weights,
            fallback_weights,
        )
        mean_confidence = confidence_sum / event_count
        aggregated = (raw_per_event * weights[:, None, :, None]).sum(dim=2)
        aggregated = aggregated * mean_confidence[:, None, :]
        embedding = self.projection(aggregated)
        valid = valid_time_mask.to(device=device, dtype=torch.bool)
        return embedding * valid[:, :, None].to(embedding.dtype)


class SensorTemporalEncoder(nn.Module):
    """M2 temporal encoder with an optional M3 observed-quality prior."""

    def __init__(
        self,
        *,
        input_channels: int,
        feature_channels: int,
        time_embedding_dim: int,
        quality_prior_strength: float = 0.75,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or feature_channels <= 0 or time_embedding_dim <= 0:
            raise ValueError("sensor encoder dimensions must be positive")
        if quality_prior_strength < 0:
            raise ValueError("quality prior strength cannot be negative")
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.time_embedding_dim = time_embedding_dim
        self.quality_prior_strength = float(quality_prior_strength)
        groups = _group_count(feature_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )
        self.time_projection = nn.Linear(
            time_embedding_dim, feature_channels, bias=False
        )
        self.attention_score = nn.Conv2d(feature_channels, 1, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor,
        time_embedding: torch.Tensor,
        valid_time_mask: torch.Tensor,
        observed_quality: torch.Tensor,
        *,
        use_quality: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 5:
            raise ValueError("sensor image must have shape [B,T,C,H,W]")
        batch, steps, channels, height, width = image.shape
        if channels != self.input_channels:
            raise ValueError("sensor input-channel count changed")
        if time_embedding.shape != (batch, steps, self.time_embedding_dim):
            raise ValueError("event-time embedding shape mismatch")
        if valid_time_mask.shape != (batch, steps):
            raise ValueError("sensor valid-time mask shape mismatch")
        if observed_quality.shape != (batch, steps, height, width):
            raise ValueError("observed quality must have shape [B,T,H,W]")
        if not bool(torch.isfinite(image).all()) or not bool(
            torch.isfinite(observed_quality).all()
        ):
            raise ValueError("sensor image/quality must be finite")
        if bool(((observed_quality < 0) | (observed_quality > 1)).any()):
            raise ValueError("observed quality must lie in [0,1]")

        valid = valid_time_mask.to(device=image.device, dtype=torch.bool)
        sensor_available = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[~sensor_available, 0] = True
        features = self.stem(image.reshape(batch * steps, channels, height, width))
        features = features.reshape(
            batch, steps, self.feature_channels, height, width
        )
        temporal_bias = self.time_projection(time_embedding).reshape(
            batch, steps, self.feature_channels, 1, 1
        )
        features = features + temporal_bias
        logits = self.attention_score(
            torch.tanh(features.reshape(batch * steps, self.feature_channels, height, width))
        ).reshape(batch, steps, height, width)
        if use_quality:
            quality_logit = torch.logit(observed_quality.clamp(1e-4, 1 - 1e-4))
            logits = logits + self.quality_prior_strength * quality_logit
        logits = logits.masked_fill(~safe_valid[:, :, None, None], -torch.inf)
        attention = torch.softmax(logits, dim=1)
        attention = attention * sensor_available[:, None, None, None].to(
            attention.dtype
        )
        pooled = (features * attention[:, :, None]).sum(dim=1)
        pooled = pooled * sensor_available[:, None, None, None].to(pooled.dtype)
        quality_summary = (observed_quality * attention).sum(dim=1).clamp(0.0, 1.0)
        quality_summary = quality_summary * sensor_available[:, None, None].to(
            quality_summary.dtype
        )
        return pooled, attention, quality_summary, sensor_available


class QualityAwareSensorSetFusion(nn.Module):
    """M2 labeled-set fusion with a hard availability mask and M3 prior."""

    def __init__(
        self,
        *,
        sensor_names: Sequence[str] = SENSOR_ORDER,
        feature_channels: int = 32,
        quality_prior_strength: float = 0.75,
        residual_logit_bound: float = 2.0,
    ) -> None:
        super().__init__()
        names = tuple(sensor_names)
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError("sensor names must be unique")
        if feature_channels <= 0 or quality_prior_strength < 0:
            raise ValueError("invalid fusion dimensions or quality strength")
        if residual_logit_bound < 0:
            raise ValueError("residual logit bound cannot be negative")
        self.sensor_names = names
        self.feature_channels = feature_channels
        self.quality_prior_strength = float(quality_prior_strength)
        self.residual_logit_bound = float(residual_logit_bound)
        self.sensor_embedding = nn.Parameter(
            torch.zeros(len(names), feature_channels)
        )
        nn.init.normal_(self.sensor_embedding, std=0.02)
        self.residual_score = nn.Conv2d(feature_channels, 1, kernel_size=1)
        nn.init.zeros_(self.residual_score.weight)
        nn.init.zeros_(self.residual_score.bias)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        observed_quality: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        *,
        use_sensor_set_fusion: bool,
        use_quality: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if set(features) != set(self.sensor_names):
            raise ValueError("feature mapping must contain the exact labeled sensor set")
        if set(observed_quality) != set(self.sensor_names):
            raise ValueError("quality mapping must contain the exact labeled sensor set")
        ordered = [features[name] for name in self.sensor_names]
        reference_shape = ordered[0].shape
        if len(reference_shape) != 4 or reference_shape[1] != self.feature_channels:
            raise ValueError("sensor features must have shape [B,C,H,W]")
        if any(value.shape != reference_shape for value in ordered):
            raise ValueError("sensor feature shapes must match")
        batch, _, height, width = reference_shape
        if availability.shape != (batch, len(self.sensor_names)):
            raise ValueError("sensor availability shape mismatch")
        availability = availability.to(device=ordered[0].device, dtype=torch.bool)
        if not bool(availability.any(dim=1).all()):
            raise ValueError("every sample must retain at least one sensor")
        stacked = torch.stack(ordered, dim=1)
        sensor_identity = self.sensor_embedding.reshape(
            1, len(self.sensor_names), self.feature_channels, 1, 1
        )
        labeled = stacked + sensor_identity
        if use_sensor_set_fusion:
            residual = self.residual_score(
                labeled.reshape(
                    batch * len(self.sensor_names),
                    self.feature_channels,
                    height,
                    width,
                )
            ).reshape(batch, len(self.sensor_names), height, width)
            logits = self.residual_logit_bound * torch.tanh(residual)
        else:
            logits = torch.zeros(
                (batch, len(self.sensor_names), height, width),
                device=stacked.device,
                dtype=stacked.dtype,
            )
        if use_quality:
            quality_stack = torch.stack(
                [observed_quality[name] for name in self.sensor_names], dim=1
            )
            if quality_stack.shape != (batch, len(self.sensor_names), height, width):
                raise ValueError("sensor quality-summary shape mismatch")
            if not bool(torch.isfinite(quality_stack).all()) or bool(
                ((quality_stack < 0) | (quality_stack > 1)).any()
            ):
                raise ValueError("sensor quality summaries must lie in [0,1]")
            logits = logits + self.quality_prior_strength * torch.logit(
                quality_stack.clamp(1e-4, 1 - 1e-4)
            )
        logits = logits.masked_fill(~availability[:, :, None, None], -torch.inf)
        weights = torch.softmax(logits, dim=1)
        fused = (labeled * weights[:, :, None]).sum(dim=1)
        return fused, weights


class StructuredSensorTimeDropout(nn.Module):
    """M4: reproducible sensor/orbit/contiguous/random observation loss."""

    def __init__(
        self,
        *,
        sensor_names: Sequence[str] = SENSOR_ORDER,
        scenario_probabilities: Mapping[str, float] | None = None,
        contiguous_fraction: float = 1 / 3,
        random_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        names = tuple(sensor_names)
        if names != SENSOR_ORDER:
            raise ValueError("M4 currently requires the fixed S1-asc/S1-dsc/S2 order")
        if scenario_probabilities is None:
            scenario_probabilities = {
                "none": 0.20,
                "drop_modality": 0.20,
                "drop_sar_orbit": 0.20,
                "drop_contiguous_time": 0.20,
                "drop_random_time": 0.20,
            }
        if set(scenario_probabilities) != set(SCENARIO_ORDER):
            raise ValueError("scenario probabilities must cover the fixed M4 cases")
        probabilities = tuple(
            float(scenario_probabilities[name]) for name in SCENARIO_ORDER
        )
        if any(value < 0 or not math.isfinite(value) for value in probabilities):
            raise ValueError("scenario probabilities must be finite and nonnegative")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("scenario probabilities must sum to one")
        if not 0 < contiguous_fraction < 1 or not 0 < random_fraction < 1:
            raise ValueError("time-drop fractions must lie in (0,1)")
        self.sensor_names = names
        self.contiguous_fraction = float(contiguous_fraction)
        self.random_fraction = float(random_fraction)
        self.register_buffer(
            "scenario_probabilities",
            torch.tensor(probabilities, dtype=torch.float64),
        )

    @staticmethod
    def _random_choice(
        candidates: torch.Tensor, *, generator: torch.Generator | None
    ) -> int:
        selected = torch.randint(
            candidates.numel(),
            (1,),
            device=candidates.device,
            generator=generator,
        )
        return int(candidates[int(selected.item())].item())

    def forward(
        self,
        valid_time_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        force_scenario: str | None = None,
    ) -> StructuredDropoutResult:
        if valid_time_mask.ndim != 3 or valid_time_mask.shape[1] != len(
            self.sensor_names
        ):
            raise ValueError("M4 valid-time mask must have shape [B,3,T]")
        mask = valid_time_mask.to(dtype=torch.bool)
        if not bool(mask.any(dim=(1, 2)).all()):
            raise ValueError("M4 input must contain at least one observation per sample")
        if force_scenario is not None and force_scenario not in SCENARIO_ORDER:
            raise ValueError("unknown forced M4 scenario")
        output = mask.clone()
        batch, _, steps = output.shape
        if force_scenario is None:
            scenario_indices = torch.multinomial(
                self.scenario_probabilities.to(device=mask.device),
                batch,
                replacement=True,
                generator=generator,
            )
            scenarios = [SCENARIO_ORDER[int(index)] for index in scenario_indices]
        else:
            scenarios = [force_scenario] * batch

        realized: list[str] = []
        for batch_index, requested in enumerate(scenarios):
            original = output[batch_index].clone()
            scenario = requested
            if scenario == "drop_modality":
                available = output[batch_index].any(dim=1).nonzero().flatten()
                if available.numel() <= 1:
                    scenario = "none"
                else:
                    sensor_index = self._random_choice(
                        available, generator=generator
                    )
                    output[batch_index, sensor_index] = False
            elif scenario == "drop_sar_orbit":
                available_sar = output[batch_index, :2].any(dim=1).nonzero().flatten()
                if available_sar.numel() == 0:
                    scenario = "none"
                else:
                    sensor_index = self._random_choice(
                        available_sar, generator=generator
                    )
                    candidate = output[batch_index].clone()
                    candidate[sensor_index] = False
                    if bool(candidate.any()):
                        output[batch_index] = candidate
                    else:
                        scenario = "none"
            elif scenario == "drop_contiguous_time":
                length = max(1, min(steps - 1, round(steps * self.contiguous_fraction)))
                start = int(
                    torch.randint(
                        steps - length + 1,
                        (1,),
                        device=mask.device,
                        generator=generator,
                    ).item()
                )
                output[batch_index, :, start : start + length] = False
            elif scenario == "drop_random_time":
                count = max(1, min(steps - 1, round(steps * self.random_fraction)))
                indices = torch.randperm(
                    steps, device=mask.device, generator=generator
                )[:count]
                output[batch_index, :, indices] = False
            if torch.equal(output[batch_index], original):
                scenario = "none"
            if not bool(output[batch_index].any()):
                output[batch_index] = original
                scenario = "none"
            realized.append(scenario)
        return StructuredDropoutResult(output, tuple(realized))


class ResidualSpatialBlock(nn.Module):
    """Small normalized residual block used only as a shared spatial backbone."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(value) + self.skip(value))


class SharedSpatialUNetHead(nn.Module):
    """Two-scale spatial context head shared unchanged by every M1--M4 ablation."""

    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0:
            raise ValueError("spatial head channels must be positive")
        self.encoder0 = ResidualSpatialBlock(input_channels, base_channels)
        self.encoder1 = ResidualSpatialBlock(base_channels, base_channels * 2)
        self.bottleneck = ResidualSpatialBlock(base_channels * 2, base_channels * 4)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder1 = ResidualSpatialBlock(base_channels * 4, base_channels * 2)
        self.up0 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder0 = ResidualSpatialBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[-2] % 4 or value.shape[-1] % 4:
            raise ValueError("spatial U-Net head requires [B,C,H,W] with H/W divisible by 4")
        level0 = self.encoder0(value)
        level1 = self.encoder1(torch.nn.functional.max_pool2d(level0, 2))
        bottleneck = self.bottleneck(torch.nn.functional.max_pool2d(level1, 2))
        decoded1 = self.decoder1(torch.cat((self.up1(bottleneck), level1), dim=1))
        decoded0 = self.decoder0(torch.cat((self.up0(decoded1), level0), dim=1))
        return self.output(decoded0)


class ObservationRobustSegmenter(nn.Module):
    """Compact end-to-end composition used to test the M1--M4 interfaces."""

    def __init__(
        self,
        *,
        sensor_channels: Mapping[str, int] | None = None,
        feature_channels: int = 32,
        time_embedding_dim: int = 16,
        quality_prior_strength: float = 0.75,
        residual_logit_bound: float = 2.0,
        dropout_probabilities: Mapping[str, float] | None = None,
        spatial_head: str = "compact",
    ) -> None:
        super().__init__()
        if sensor_channels is None:
            sensor_channels = {"s1asc": 2, "s1dsc": 2, "s2": 10}
        if tuple(sensor_channels) != SENSOR_ORDER:
            raise ValueError("sensor channel mapping must use the fixed labeled order")
        if feature_channels <= 0 or time_embedding_dim <= 0:
            raise ValueError("method widths must be positive")
        self.sensor_channels = dict(sensor_channels)
        self.feature_channels = feature_channels
        self.time_embedding_dim = time_embedding_dim
        self.event_time_encoder = EventRelativeTimeEncoder(
            embedding_dim=time_embedding_dim
        )
        self.sensor_encoders = nn.ModuleDict(
            {
                name: SensorTemporalEncoder(
                    input_channels=self.sensor_channels[name],
                    feature_channels=feature_channels,
                    time_embedding_dim=time_embedding_dim,
                    quality_prior_strength=quality_prior_strength,
                )
                for name in SENSOR_ORDER
            }
        )
        self.sensor_fusion = QualityAwareSensorSetFusion(
            sensor_names=SENSOR_ORDER,
            feature_channels=feature_channels,
            quality_prior_strength=quality_prior_strength,
            residual_logit_bound=residual_logit_bound,
        )
        dem_channels = max(8, feature_channels // 2)
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, dem_channels, 3, padding=1),
            nn.GroupNorm(_group_count(dem_channels), dem_channels),
            nn.GELU(),
            nn.Conv2d(dem_channels, dem_channels, 3, padding=1),
            nn.GroupNorm(_group_count(dem_channels), dem_channels),
            nn.GELU(),
        )
        if spatial_head == "compact":
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(feature_channels + dem_channels, feature_channels, 3, padding=1),
                nn.GroupNorm(_group_count(feature_channels), feature_channels),
                nn.GELU(),
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(feature_channels, 1, kernel_size=1),
            )
        elif spatial_head == "shared_unet":
            self.segmentation_head = SharedSpatialUNetHead(
                feature_channels + dem_channels, feature_channels
            )
        else:
            raise ValueError("unknown shared spatial head")
        self.spatial_head = spatial_head
        self.structured_dropout = StructuredSensorTimeDropout(
            scenario_probabilities=dropout_probabilities
        )

    def forward(
        self,
        *,
        sensor_images: Mapping[str, torch.Tensor],
        acquisition_days: Mapping[str, torch.Tensor],
        valid_time_masks: Mapping[str, torch.Tensor],
        observed_quality: Mapping[str, torch.Tensor],
        dem: torch.Tensor,
        event_days: torch.Tensor,
        event_confidence: torch.Tensor,
        event_mask: torch.Tensor,
        ablation: str = "A4",
        dropout_generator: torch.Generator | None = None,
        force_dropout_scenario: str | None = None,
        return_auxiliary: bool = False,
    ) -> dict[str, object]:
        if ablation not in ABLATION_SPECS:
            raise ValueError("unknown M1--M4 ablation")
        if (
            set(sensor_images) != set(SENSOR_ORDER)
            or set(acquisition_days) != set(SENSOR_ORDER)
            or set(valid_time_masks) != set(SENSOR_ORDER)
            or set(observed_quality) != set(SENSOR_ORDER)
        ):
            raise ValueError("all sensor mappings must contain the exact labeled set")
        spec = ABLATION_SPECS[ablation]
        reference = sensor_images[SENSOR_ORDER[0]]
        if reference.ndim != 5:
            raise ValueError("sensor images must have shape [B,T,C,H,W]")
        batch, steps, _, height, width = reference.shape
        if any(sensor_images[name].shape[0] != batch for name in SENSOR_ORDER):
            raise ValueError("sensor batches must match")
        if any(sensor_images[name].shape[1] != steps for name in SENSOR_ORDER):
            raise ValueError("current method gate requires a common padded time length")
        if any(sensor_images[name].shape[-2:] != (height, width) for name in SENSOR_ORDER):
            raise ValueError("sensor spatial shapes must match")
        if dem.shape != (batch, 1, height, width):
            raise ValueError("DEM must have shape [B,1,H,W]")
        if not bool(torch.isfinite(dem).all()):
            raise ValueError("DEM must be finite")

        stacked_valid = torch.stack(
            [valid_time_masks[name].to(dtype=torch.bool) for name in SENSOR_ORDER],
            dim=1,
        )
        if stacked_valid.shape != (batch, len(SENSOR_ORDER), steps):
            raise ValueError("valid-time masks must all have shape [B,T]")
        scenarios = ("none",) * batch
        should_apply_dropout = spec.use_structured_dropout and (
            self.training or force_dropout_scenario is not None
        )
        if should_apply_dropout:
            dropout = self.structured_dropout(
                stacked_valid,
                generator=dropout_generator,
                force_scenario=force_dropout_scenario,
            )
            stacked_valid = dropout.valid_time_mask
            scenarios = dropout.scenarios

        encoded_features: dict[str, torch.Tensor] = {}
        quality_summaries: dict[str, torch.Tensor] = {}
        temporal_attention: dict[str, torch.Tensor] = {}
        availability_rows: list[torch.Tensor] = []
        for sensor_index, name in enumerate(SENSOR_ORDER):
            valid = stacked_valid[:, sensor_index]
            if spec.use_event_time:
                time_embedding = self.event_time_encoder(
                    acquisition_days[name],
                    valid,
                    event_days,
                    event_confidence,
                    event_mask,
                )
            else:
                time_embedding = torch.zeros(
                    (batch, steps, self.time_embedding_dim),
                    device=sensor_images[name].device,
                    dtype=sensor_images[name].dtype,
                )
            feature, attention, quality, available = self.sensor_encoders[name](
                sensor_images[name],
                time_embedding,
                valid,
                observed_quality[name],
                use_quality=spec.use_quality,
            )
            encoded_features[name] = feature
            temporal_attention[name] = attention
            quality_summaries[name] = quality
            availability_rows.append(available)
        availability = torch.stack(availability_rows, dim=1)
        fused, sensor_weights = self.sensor_fusion(
            encoded_features,
            quality_summaries,
            availability,
            use_sensor_set_fusion=spec.use_sensor_set_fusion,
            use_quality=spec.use_quality,
        )
        logits = self.segmentation_head(torch.cat((fused, self.dem_encoder(dem)), dim=1))
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("observation-robust method produced non-finite logits")
        result: dict[str, object] = {"segmentation": logits}
        if return_auxiliary:
            result.update(
                {
                    "sensor_weights": sensor_weights,
                    "temporal_attention": temporal_attention,
                    "sensor_availability": availability,
                    "dropout_scenarios": scenarios,
                    "ablation": ablation,
                }
            )
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
