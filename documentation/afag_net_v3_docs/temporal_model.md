# models/temporal/temporal_model.py

## Purpose

`models/temporal/temporal_model.py` implements the temporal aggregation module for AFAGNet v3. It converts fused per-frame feature maps into a single video-level representation.

## Inputs

- `Ffusion`: `(B, N, C, H, W)` — fused branch features for a sequence of frames

## Output

- `video_feat`: `(B, embed_dim)` — video-level embedding used for classification

## Architectural components

- Linear projection from `in_channels` to `embed_dim`
- Layer norm on the projected sequence
- Learned positional embeddings
- Transformer encoder with `num_layers` and `num_heads`
- Learnable temporal weights `w1`, `w2`, `w4`

## Forward pass

### 1. Spatial attention mask

- `get_spatial_mask(x)` computes an energy map from absolute feature values.
- It normalizes each frame independently to create a soft attention mask.

### 2. Masked global pooling

- `global_pool_masked(x, mask)` performs weighted pooling over spatial dimensions.
- The result `Ft` is `(B, N, C)`.

### 3. Temporal difference features

- `compute_differences(Ft)` computes:
  - `D1`: absolute differences between adjacent frames
  - `D2`: absolute differences between frames two steps apart
  - `D4`: absolute differences between frames four steps apart
- Differences are padded to preserve sequence length.

### 4. Sequence construction

- The final sequence is concatenated as:
  - `[Ft, 0.5 * w1 * D1, 0.3 * w2 * D2, 0.2 * w4 * D4]`
- This gives a sequence length of `4N`.

### 5. Projection and positional embedding

- The concatenated sequence is projected to `embed_dim`.
- Learned positional embeddings are added.
- A motion-aware bias based on `D1` importance is also added.

### 6. Transformer encoding

- The sequence passes through a `TransformerEncoder`.
- Output is aggregated by mean pooling over the time dimension.


