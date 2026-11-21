# LandslideObsRobust

Supporting materials for **Robustness Remains Regime-Dependent in Event-Aligned Multimodal Landslide Segmentation under Missing Satellite Observations**.

## Core model

- `model/observation_robust.py`: event-relative temporal encoding, sensor-identity-aware fusion, pixel-validity priors, structured sensor-time dropout, and the spatial segmentation head.
- `model/observation_robust_deterministic.py`: deterministic structured-dropout sampler used with the model.

The implementation requires Python 3.10 or later and PyTorch. Instantiate the A4 architecture as follows:

```python
from model.observation_robust_deterministic import DeterministicObservationRobustSegmenter

model = DeterministicObservationRobustSegmenter(spatial_head="shared_unet")
```

Use `ablation="A4"` in `forward`. The model takes labeled Sentinel-1 ascending, Sentinel-1 descending, and Sentinel-2 inputs, acquisition days, valid-time masks, pixel-validity maps, a DEM, and event-date metadata. See the `forward` signature for tensor inputs. Event confidence represents the sample-level mean broadcast to each valid event date. A4 has 539,125 trainable parameters with the settings above.

## Data splits

`splits/train.jsonl`, `splits/val.jsonl`, and `splits/test.jsonl` identify the Sen12Landslides training, validation, and test samples and their event/spatial groups. They contain 2,737, 673, and 802 records, respectively, in 112, 10, and 14 retained groups. The 145 groups reported by the partition builder are counted before the spatial guard. The public copies retain the split assignments, sample identifiers, group identifiers, source file names, and file checksums; machine-specific absolute paths were removed. The original data are available at https://huggingface.co/datasets/paulhoehn/Sen12Landslides (CC BY 4.0; DOI: https://doi.org/10.57967/hf/5883).

## Evaluation summary

`audit_summary.json` contains the principal evaluation values and selected audit information. The external DLR/GFZ data record is available at https://doi.org/10.5281/zenodo.17007637.

`SHA256SUMS` lists checksums for the released files.
