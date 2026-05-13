# models/branches/frequency_branch_raw.py

## Purpose

`models/branches/frequency_branch_raw.py` implements the frequency branch for AFAGNet v3. It extracts multi-scale spectral features from raw input frames using FFT-based analysis and attention mechanisms.

## Components

### `MultiScaleFFT`

#### Purpose

- Computes frequency representations at multiple block scales.
- Produces both magnitude and phase feature maps for each scale.
- Operates in `float32` internally to avoid FFT overflow and NaN values.

#### Key steps

1. Convert input to `float32`.
2. For each block size in `(2, 4, 7)`:
   - Crop the input to dimensions divisible by the block size.
   - Reshape the input into non-overlapping blocks.
   - Apply `torch.fft.rfft2` to compute the real-valued FFT.
   - Compute log-magnitude via `log1p(abs(fft))`.
   - Compute normalized phase via `angle(fft)` scaled to `[0,1]`.
   - Reassemble the frequency outputs into the original spatial layout.
   - Interpolate back to the original resolution if cropping occurred.
3. Project magnitude and phase streams with separate `1×1` convolutions.
4. Fuse all scales into a single output tensor.

#### Output

- Returns a fused tensor with shape `(B, channels, H, W)`.

### `FrequencyBranch`

#### Purpose

- Converts raw 6-channel input into a frequency feature tensor suitable for fusion with the spatial and noise branches.

#### Pipeline

1. Raw adapter: converts input to `256` channels and downsamples from `224×224` to `28×28`.
2. Spatial normalization per-channel.
3. Multi-scale FFT processing.
4. Post-FFT feature refinement with convolution, batch norm, and GELU.
5. Channel attention using both average and max pooling.
6. Spatial attention via concatenated average/max projections.
7. Residual projection and final output.

#### Output

- `(B*N, 256, 14, 14)`

