import torch
import torch.nn as nn


def get_srm_kernels():
    # 3 basic SRM filters (simplified)
    k1 = [[0, 0, 0],
          [0, 1, -1],
          [0, 0, 0]]

    k2 = [[0, 0, 0],
          [1, -2, 1],
          [0, 0, 0]]

    k3 = [[-1, 2, -1],
          [2, -4, 2],
          [-1, 2, -1]]

    kernels = torch.tensor([k1, k2, k3], dtype=torch.float32)
    return kernels.unsqueeze(1)  # (3,1,3,3)


class NoiseBranch(nn.Module):
    def __init__(self, channels=256):
        super().__init__()

        srm_kernels = get_srm_kernels()

        self.srm = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=channels
        )

        # 2. LEARNABLE CONV PATH (The 'Hybrid' part)
        # We use a depthwise conv here to match the SRM grouping
        self.learnable_path = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

        # Initialize every depthwise channel with the same averaged SRM prior.
        with torch.no_grad():
            kernel = srm_kernels.mean(dim=0)  # (1, 1, 3, 3)
            self.srm.weight.copy_(kernel.repeat(channels, 1, 1, 1))

        # Keep the forensic prior fixed and let the learnable path adapt.
        for p in self.srm.parameters():
            p.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):            ### here we get backbone output for improve performance we use raw input
        # Combine the fixed forensic prior with the learnable path
        # This fulfills the Hybrid Noise Learning criteria
        res = self.srm(x) + self.learnable_path(x)
        x = self.conv(res)
        return x
