# models/branches/spatial_branch.py

## Purpose

`models/branches/spatial_branch.py` defines the spatial branch of AFAGNet v3. It combines fine-grained local texture and coarse semantic context using a dual-path architecture.

## Components

### `CBAM`

- Channel and spatial attention module.
- Applies channel attention via global average pooling and a small MLP.
- Applies spatial attention by computing average and max projections across channels and using a `7×7` convolution.
- Outputs an attention-weighted feature map.

### `SpatialBranch`

#### Inputs

- `F_low`: `(B*N, 256, H, W)` 
- `F_high`: `(B*N, 256, H/2, W/2)`

#### Internal structure

- Path A processes `F_low`:
  - High-pass filter via `Conv2d(channels, channels, 3, padding=1, bias=False)`.
  - Residual connection with the input.
  - Convolution + BatchNorm + ReLU.
  - `CBAM` attention.

- Path B processes `F_high`:
  - Projects `F_high` channel count to match `F_low`.
  - Applies another high-pass filter and residual connection.
  - Convolution + BatchNorm + ReLU.
  - `CBAM` attention.
  - Upsamples to `F_low` spatial size using bilinear interpolation.

- Fusion stage:
  - Concatenates path outputs along channels.
  - Reduces the fused tensor back to `channels` using a `1×1` convolution.

#### Output

- `fused`: `(B*N, 256, H, W)` — a single spatial feature map combining local and semantic cues.
