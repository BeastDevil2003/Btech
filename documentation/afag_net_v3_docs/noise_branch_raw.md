# models/branches/noise_branch_raw.py

## Purpose

`models/branches/noise_branch_raw.py` defines the noise branch for AFAGNet v3. It extracts camera noise residuals using SRM filters and combines them with learned convolutional features.

## Components

### `get_srm_kernels()`

- Returns a tensor containing three SRM kernels.
- These are fixed, non-learned filters that detect first- and second-order residual patterns in images.
- The current default is the 3-kernel version for VRAM-efficient training.

### `NoiseBranch`

#### Inputs

- Raw 6-channel input: `(B*N, 6, 224, 224)`.

#### Pipeline

1. Adapter: converts raw input into `256` channels with three stride-2 convolutions.
   - Reduces resolution from `224×224` to `28×28`.
2. Instance normalization for stability with small batch sizes.
3. Fixed SRM filtering with depthwise convolution.
4. Learnable complementary noise extraction path.
5. Convolutional refinement.
6. Attention gating to emphasize important noise features.
7. Residual addition of the adapter output.

#### Output

- `(B*N, 256, 14, 14)`
