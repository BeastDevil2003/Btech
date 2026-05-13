import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """Channel + Spatial attention. Tells the network 'what' and 'where' to look."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        x = x * y
        avg  = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        s = self.spatial(torch.cat([avg, maxv], dim=1))
        return x * s


class SpatialBranch(nn.Module):
    def __init__(self, channels=256, high_channels=512):
        super().__init__()

        self.high_pass_a = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv_a = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.cbam_a = CBAM(channels)

        self.proj_b = nn.Conv2d(high_channels, channels, 1, bias=False)
        self.high_pass_b = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv_b = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.cbam_b = CBAM(channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, F_low, F_high=None):
        x_a = self.high_pass_a(F_low) + F_low
        x_a = self.conv_a(x_a)
        x_a = self.cbam_a(x_a)

        if F_high is None:
            return x_a

        x_b = self.proj_b(F_high)
        x_b = self.high_pass_b(x_b) + x_b
        x_b = self.conv_b(x_b)
        x_b = self.cbam_b(x_b)
        x_b = F.interpolate(x_b, size=F_low.shape[2:], mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([x_a, x_b], dim=1))
        return fused