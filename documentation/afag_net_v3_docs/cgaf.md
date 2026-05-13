# models/fusion/cgaf.py

## Purpose

`models/fusion/cgaf.py` implements `CGAFv2`, the Cross-branch Gated Adaptive Fusion module used to combine spatial, frequency, and noise features.

## Inputs

- `Fs`: spatial branch features, `(B*N, 256, 14, 14)`
- `Ff`: frequency branch features, `(B*N, 256, 14, 14)`
- `Fn`: noise branch features, `(B*N, 256, 14, 14)`
- `F_high`: optional high-level features, `(B*N, 256, 7, 7)`

## Output

- `Ffusion`: fused feature tensor, `(B*N, 256, 14, 14)`
- `Csf`, `Csn`, `Cfn`: pairwise similarity maps used for consistency and gating

## Core concepts

### Bounded temperature

- `log_temp` is learned and exponentiated to produce a temperature in `[0.1, 10.0]`.
- This prevents fusion from becoming too sharp (dead gradients) or too flat.

### Similarity maps

- `Csf`: similarity between spatial and frequency features
- `Csn`: similarity between spatial and noise features
- `Cfn`: similarity between frequency and noise features

Similarity is computed by normalizing feature vectors and summing their elementwise product.

### Gating

- `Wf = sigmoid(Csf * temp)`
- `Wn = sigmoid(Csn * temp)`
- The weights are normalized and modulated by `cross_gate = sigmoid(Cfn)`.

### F_high context

- If `use_f_high` is enabled, a pooled `F_high` vector is passed through a small gate network.
- This gate scales `Wf` and `Wn` to incorporate global semantic context.

### Learnable residual weight

- `alpha_raw` is transformed with `softplus` to ensure a positive scaling factor.
- The final fused features are `Fs + alpha * (Wf * Ff + Wn * Fn)`.

### Refinement block

- A small convolutional residual block refines the fused output before returning.

## Additional function

### `consistency_loss(Fs, Ff, Fn)`

- Returns the L1 distance between branch feature maps.
- Encourages all three branches to agree on spatial structure.


