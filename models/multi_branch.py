from models.branches.spatial_branch import SpatialBranch
from models.branches.frequency_branch import FrequencyBranch
from models.branches.noise_branch import NoiseBranch
import torch.nn as nn


class MultiBranchModule(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.spatial = SpatialBranch(channels)
        self.frequency = FrequencyBranch(channels)
        self.noise = NoiseBranch(channels)

    def forward(self, x):
        Fs = self.spatial(x)
        Ff = self.frequency(x)
        Fn = self.noise(x)

        return Fs, Ff, Fn