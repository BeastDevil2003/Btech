# models/backbone.py

## Purpose

`models/backbone.py` implements the visual backbone used by AFAGNet v3. It converts 6-channel input frames into two feature maps:

- `F_low`: fine-grained texture features at `28×28`
- `F_high`: coarse semantic features at `7×7`

The file also provides an optional EfficientNet-B4 backbone.( i try to write but not tested it working error free or not)

## Classes

### `AFAGBackbone`


- Accept a 6-channel input tensor: `(B*N, 6, 224, 224)`.
- Adapt 6 channels to 3 channels using a learnable `1×1` convolution.
- Normalize adapted RGB input with ImageNet mean/std.
- Extract multi-scale features with a pretrained `timm` backbone in `features_only=True` mode.
- Project selected feature maps to 256 channels.

#### Key design decisions

- `input_adapter` is initialized as identity for the first 3 channels, so the backbone initially receives RGB-only inputs and learns to incorporate YCbCr gradually.
- Normalization is applied after `input_adapter` and before the backbone, correcting a v1 mistake where normalized unnatural input hurt convergence.
- The code uses `features[-3]` as `F_low` and `features[-1]` as `F_high` for the `mobilevitv2_100` backbone.
- `F_low` is projected from `192 → 256` and `F_high` from `512 → 256`.
- `F_high` is interpolated to `(7, 7)` before projection for consistency.

#### Output shapes

- `F_low`: `(B*N, 256, 28, 28)`
- `F_high`: `(B*N, 256, 7, 7)`

#### Additional methods

- `_detect_feature_channels()` probes the backbone to infer output channel counts, with a fallback for MobileViT.
- `freeze_backbone()` and `unfreeze_backbone()` control backbone trainability for AltFreezing strategies.

### `EfficientNetBackbone`


An alternative backbone with higher accuracy at the cost of significantly more VRAM.

#### Differences from `AFAGBackbone`

- Uses `efficientnet_b4` in `features_only` mode.
- Projects stage outputs from `48 → 256` and `272 → 256`.
- Resizes feature maps to the expected shapes:
  - `F_low`: `(14, 14)` → projected to `256` channels
  - `F_high`: `(7, 7)`
- Recommended only for GPUs with ≥6GB VRAM.

### `build_backbone(backbone_type, pretrained)`

- Returns `EfficientNetBackbone` when `backbone_type == "efficientnet"`.
- Otherwise returns the default `AFAGBackbone`.


