# models/afag_net_v3.py

## Purpose

`models/afag_net_v3.py` defines the full AFAGNet v3 architecture for deepfake detection. It assembles the backbone, three feature branches, fusion module, temporal model, and output heads into a single end-to-end network.

## Main components

- `AFAGBackbone` — extracts low- and high-resolution visual features from 6-channel input.
- `SpatialBranch` — combines fine texture and coarse semantic features in a dual-path spatial module.
- `FrequencyBranch` — computes multi-scale FFT-based frequency features from raw input.
- `NoiseBranch` — extracts noise residual features using SRM kernels and learned filtering.
- `CGAFv2` — fuses the three branch outputs with gated adaptive fusion and optional high-level context.
- `TemporalModel` — aggregates fused per-frame features into a video-level representation.
- `OutputHeads` — produces classification logits and localization masks.

## Data flow

1. Input shape: `(B, N, 6, 224, 224)`.
2. Frames are flattened to `(B*N, 6, 224, 224)` before backbone and branch processing.
3. `AFAGBackbone` returns:
   - `F_low`: `(B*N, 256, 28, 28)`
   - `F_high`: `(B*N, 256, 7, 7)`
4. `SpatialBranch` receives `F_low` and `F_high` and outputs fused spatial features.
5. `FrequencyBranch` and `NoiseBranch` process the raw input directly and output `256×28×28` feature maps.
6. `CGAFv2` fuses the three branch outputs into `Ffusion`: `(B*N, 256, 14, 14)` and also returns similarity maps.
7. `TemporalModel` processes the fused sequence reshaped to `(B, N, 256, 14, 14)` and returns `video_feat`: `(B, 512)`.
8. `OutputHeads` consumes `video_feat`, `Ffusion`, and `F_high` to produce:
   - `pred_cls`: `(B, 1)` classification logit
   - `pred_mask`: `(B, 1, 224, 224)` localization mask
   - `mask_sequence`, `confidence`, and `stability` explainability signals.

## Key classes

### `ClassificationHead`

- Uses a gated path on `high_feat` to control how much high-level semantic information influences classification.
- Concatenates gated high-level features with temporal features.
- Applies LayerNorm, dropout, and MLP layers to output a final logit.

### `LocalizationHead`

- Applies an initial convolution to fused features.
- Optionally upsamples `F_high` and merges it with the decoder input.
- Uses a 3-step transposed convolution decoder to produce a `224×224` mask.
- Applies `sigmoid` before returning the mask.

### `OutputHeads`

- Pools `F_high` spatially and averages across frames for classification.
- Uses `LocalizationHead` to compute per-frame masks.
- Selects the middle frame mask as the predicted localization output.
- Computes confidence and stability signals from the mask sequence.
- Produces:
  - `pred_cls`: classification logit.
  - `pred_mask`: generated mask for the center frame.
  - `mask_sequence`: per-frame mask predictions.
  - `confidence`: confidence map from mask predictions.
  - `stability`: temporal mask stability.

### `AFAGNetV3`

- Creates all component modules.
  - `AFAGBackbone()`
  - `SpatialBranch(channels=256, high_channels=256)`
  - `FrequencyBranch()`
  - `NoiseBranch()`
  - `CGAFv2(channels=256, use_f_high=True)`
  - `TemporalModel()`
  - `OutputHeads()`
- Uses `SpatialBranch(channels=256, high_channels=256)` to match the backbone-projected `F_high`.
- Uses `CGAFv2(channels=256, use_f_high=True)` to include high-level gating.
- In forward pass:
  1. Computes backbone features.
  2. Runs each branch.
  3. Casts all branch outputs to backbone dtype for AMP safety.
  4. Fuses branches.
  5. Runs temporal aggregation.
  6. Produces classification and mask outputs.

`AFAGNetV3.forward()` returns a dictionary containing:

  - `pred_cls`, `pred_mask`, `mask_sequence`, `confidence`, `stability`
  - `F_low`, `F_high`, `Fs`, `Ff`, `Fn`
  - `Ffusion`, `Ffusion_seq`, `Csf`, `Csn`, `Cfn`

## Direct dependencies

- `models/backbone.py`
- `models/branches/spatial_branch.py`
- `models/branches/frequency_branch_raw.py`
- `models/branches/noise_branch_raw.py`
- `models/fusion/cgaf.py`
- `models/temporal/temporal_model.py`

