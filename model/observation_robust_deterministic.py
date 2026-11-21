"""Deterministic categorical M4 adapter for strict CUDA reproducibility."""

from __future__ import annotations

from typing import Mapping

import torch

from .observation_robust import (
    SCENARIO_ORDER,
    ObservationRobustSegmenter,
    StructuredDropoutResult,
    StructuredSensorTimeDropout,
)


class DeterministicStructuredSensorTimeDropout(StructuredSensorTimeDropout):
    """Replace only CUDA multinomial sampling while preserving every M4 rule."""

    def _sample_scenarios(
        self,
        batch: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[str, ...]:
        if batch <= 0:
            raise ValueError("M4 batch must be positive")
        draws = torch.rand(
            batch,
            device=device,
            dtype=torch.float64,
            generator=generator,
        ).detach().cpu().tolist()
        probabilities = self.scenario_probabilities.detach().cpu().tolist()
        scenarios: list[str] = []
        for draw in draws:
            cumulative = 0.0
            selected = SCENARIO_ORDER[-1]
            for scenario, probability in zip(SCENARIO_ORDER, probabilities):
                cumulative += float(probability)
                if float(draw) < cumulative:
                    selected = scenario
                    break
            scenarios.append(selected)
        return tuple(scenarios)

    def forward(
        self,
        valid_time_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        force_scenario: str | None = None,
    ) -> StructuredDropoutResult:
        if force_scenario is not None:
            return super().forward(
                valid_time_mask,
                generator=generator,
                force_scenario=force_scenario,
            )
        if valid_time_mask.ndim != 3:
            raise ValueError("M4 valid-time mask must have shape [B,3,T]")
        requested = self._sample_scenarios(
            int(valid_time_mask.shape[0]),
            device=valid_time_mask.device,
            generator=generator,
        )
        masks: list[torch.Tensor] = []
        realized: list[str] = []
        for index, scenario in enumerate(requested):
            result = super().forward(
                valid_time_mask[index : index + 1],
                generator=generator,
                force_scenario=scenario,
            )
            masks.append(result.valid_time_mask)
            realized.extend(result.scenarios)
        return StructuredDropoutResult(torch.cat(masks, dim=0), tuple(realized))


class DeterministicObservationRobustSegmenter(ObservationRobustSegmenter):
    """Identical A0--A4 model with a deterministic categorical M4 sampler."""

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
        super().__init__(
            sensor_channels=sensor_channels,
            feature_channels=feature_channels,
            time_embedding_dim=time_embedding_dim,
            quality_prior_strength=quality_prior_strength,
            residual_logit_bound=residual_logit_bound,
            dropout_probabilities=dropout_probabilities,
            spatial_head=spatial_head,
        )
        frozen = self.structured_dropout
        probabilities = {
            name: float(frozen.scenario_probabilities[index].item())
            for index, name in enumerate(SCENARIO_ORDER)
        }
        self.structured_dropout = DeterministicStructuredSensorTimeDropout(
            sensor_names=frozen.sensor_names,
            scenario_probabilities=probabilities,
            contiguous_fraction=frozen.contiguous_fraction,
            random_fraction=frozen.random_fraction,
        )
