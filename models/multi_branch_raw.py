import torch
import torch.nn as nn

from models.branches.spatial_branch import SpatialBranch
from models.branches.frequency_branch_raw import FrequencyBranch
from models.branches.noise_branch_raw import NoiseBranch


class MultiBranchModule(nn.Module):
    """
    Three-branch forensic feature extraction.

    Spatial  (SpatialBranch)   — dual-path using F_low + F_high (GenConViT-style)
    Frequency (FrequencyBranch) — block-DCT on raw frames
    Noise    (NoiseBranch)     — SRM-based on raw frames

    F_high is now actively used in the spatial branch instead of being discarded.
    This integrates backbone high-level semantic features into the spatial forensic cue.
    """
    def __init__(self):
        super().__init__()
        self.spatial   = SpatialBranch()
        self.frequency = FrequencyBranch()
        self.noise     = NoiseBranch()

    def forward(self, F_low, raw_frames, F_high=None):
        """
        F_low      : (B*N, 256, 14, 14)  backbone low-level features
        raw_frames : (B*N, 6,  224, 224) original 6-channel input
        F_high     : (B*N, 512, 7,  7)   backbone high-level features (optional)
        """
        Fs = self.spatial(F_low, F_high)     # dual-path if F_high provided
        Ff = self.frequency(raw_frames)
        Fn = self.noise(raw_frames)
        return Fs, Ff, Fn
