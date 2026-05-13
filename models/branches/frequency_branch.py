import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBranch(nn.Module):
    def __init__(self, channels=256, block_size=2):     #Ablation Study:b=2 vs b=4 vs b=8
        super().__init__()

        self.block_size = block_size

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # Global Channel Attention for "Adaptive Frequency Learning"
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Squeeze spatial, keep frequency channels
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

    # ---------------------------
    # BLOCK-WISE DCT (approx)
    # ---------------------------
    def block_dct(self, x):         ### here we get backbone output for improve performance we use raw input
        B, C, H, W = x.shape
        b = self.block_size

        assert H % b == 0 and W % b == 0, "H and W must be divisible by block size"

        # Step 1: Extract blocks
        x = x.unfold(2, b, b).unfold(3, b, b)  
        # (B, C, H//b, W//b, b, b)

        # Step 2: Apply FFT (DCT approx)
        x = torch.fft.fft2(x)
        x = torch.abs(x)

        # Step 3: Restore spatial layout CORRECTLY
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        # (B, C, H//b, b, W//b, b)

        x = x.view(B, C, H, W)

        return x

    def forward(self, x):
        freq = self.block_dct(x)

        out = self.conv(freq)
        attn = self.freq_gate(out)

        return out * attn