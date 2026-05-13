# AFAGNet v3 Documentation

This folder contains detailed, file-level explanations for `models/afag_net_v3.py` and every module it directly or indirectly depends on.

Files included:

- `afag_net_v3.md` — top-level model architecture and main classes
- `backbone.md` — AFAGBackbone, EfficientNetBackbone, input adapter, and feature projection
- `spatial_branch.md` — dual-path spatial feature extraction and CBAM fusion
- `frequency_branch_raw.md` — multi-scale FFT frequency branch with magnitude and phase
- `noise_branch_raw.md` — SRM-based noise residual branch and learnable attention
- `cgaf.md` — cross-branch gated adaptive fusion with bounded temperature
- `temporal_model.md` — temporal modeling, masked pooling, and transformer aggregation

Each file explains:
- Purpose and role in AFAGNet v3
- Input and output shapes
- Core computations and design decisions
- Important fixes and improvements over prior versions

