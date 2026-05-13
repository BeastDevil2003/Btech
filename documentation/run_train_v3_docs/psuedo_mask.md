# psuedo_mask.py

## Purpose

`train/psuedo_mask.py` generates pseudo localization masks for samples that lack ground-truth masks. These pseudo masks help train the localization branch and improve prediction quality for unlabelled fake samples.

## Main functions

### `_temporal_ema(cam, decay)`

- Smooths activation maps over time using exponential moving average.
- Helps enforce temporal consistency in CAM-based pseudo masks.

### `compute_gradcam_stable(model, frames, decay=0.9)`

- Computes gradients of the classification score with respect to video frames.
- Produces a stable gradient-based class activation map.
- Normalizes the CAM and applies temporal smoothing.

### `frequency_map(Ff)`

- Converts frequency branch activations to a spatial map.
- Averages absolute values across channels.
- Resizes output to `224x224`.

### `noise_map(Fn)`

- Converts noise branch activations to a spatial map.
- Averages absolute values across channels.
- Resizes output to `224x224`.

### `generate_pseudo_mask(model, frames, Fs, Ff, Fn, alpha=0.5, beta=0.3, gamma=0.2, use_gradcam=True)`

- Produces a pseudo mask from branch activations and optional GradCAM.
- If `use_gradcam=True`, combines:
  - CAM signal (`alpha`)
  - frequency map (`beta`)
  - noise map (`gamma`)
- If `use_gradcam=False`, uses frequency + noise only.
- Normalizes the output to `[0, 1]`.

## Inputs

- `model`: AFAGNetV3 model instance.
- `frames`: input video tensor `(B, N, C, H, W)`.
- `Fs`, `Ff`, `Fn`: intermediate branch feature tensors.
- `alpha`, `beta`, `gamma`: weighting coefficients.
- `use_gradcam`: whether to compute GradCAM.

## Output

- A normalized pseudo mask tensor of shape `(B, N, 1, 224, 224)` or `(B, 1, 224, 224)` depending on how it is updated later.

## Role in training

- Used by `train/train_pipeline_v3.py` during `train_one_epoch()`.
- Provides supervision on samples with no explicit mask annotation.
- Improves localization loss by guiding the model with branch activation cues.

## Notes

- `use_gradcam=True` is more informative but uses more GPU memory.
- The file includes safety guidance: do not use GradCAM within `torch.no_grad()`.
