"""
frequency_branch_v2.py  — NaN-safe, phase-aware, multi-scale frequency branch
==============================================================================
Changes from v1:
  FIX-2:  Use log1p(magnitude) — log scale gives better gradient behaviour, prevents overflow
  NEW-1:  Phase stream — phase discontinuities are strong deepfake indicators
           (GAN models produce systematic phase artifacts; blending creates phase boundaries)
  NEW-2:  Multi-scale frequency analysis (block_size 2, 4, 8)
           — different scales capture different artifact types
           — small blocks: pixel-level GAN noise; larger blocks: texture/blending artifacts
  NEW-3:  GELU activation (smoother gradient landscape than ReLU)
  NEW-4:  Spatial normalization before FFT (per-spatial-location, not global)
  NEW-5:  Frequency band attention using both channel and spatial dimensions
  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFFT(nn.Module):
    """
    Fuses magnitude (global artifact energy) and phase (structural inconsistency).
    
    Block size 2  → pixel-level GAN noise patterns
    Block size 4  → local texture artifacts
    Block size 8  → medium-scale blending boundaries
    """
    def __init__(self, channels: int, block_sizes=(2, 4, 8)):
        super().__init__()
        self.block_sizes = block_sizes
        n_scales = len(block_sizes)

        self.mag_projs = nn.ModuleList([
            nn.Conv2d(channels, channels // 2, 1) for _ in block_sizes
        ])

        self.phase_projs = nn.ModuleList([
            nn.Conv2d(channels, channels // 4, 1) for _ in block_sizes
        ])

        in_fuse = (channels // 2 + channels // 4) * n_scales
        self.fuse = nn.Sequential(
            nn.Conv2d(in_fuse, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def _block_fft(self, x: torch.Tensor, block_size: int):
        """
        Block-wise FFT: divides feature map into non-overlapping blocks,
        computes FFT within each block, returns log-magnitude and normalized phase.
        """
        B, C, H, W = x.shape
        b = block_size

        H_ = H - (H % b)
        W_ = W - (W % b)
        if H_ < b or W_ < b:
            H_, W_ = H, W
            b = max(1, min(H_, W_))

        x = x[:, :, :H_, :W_]

        x_blocks = x.unfold(2, b, b).unfold(3, b, b)

        fft = torch.fft.rfft2(x_blocks, norm='ortho')

        # log1p prevents overflow and gives better gradient scale
        mag = torch.log1p(torch.abs(fft))               # stable log magnitude

        phase = torch.angle(fft)                        
        phase = (phase + torch.pi) / (2.0 * torch.pi) 

        pad_w = b - fft.shape[-1]
        if pad_w > 0:
            mag   = F.pad(mag,   (0, pad_w))
            phase = F.pad(phase, (0, pad_w))

        mag   = mag.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H_, W_)
        phase = phase.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H_, W_)


        if H_ < H or W_ < W:
            mag   = F.interpolate(mag,   size=(H, W), mode='bilinear', align_corners=False)
            phase = F.interpolate(phase, size=(H, W), mode='bilinear', align_corners=False)

        return mag, phase

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x_f32 = x.float()   

        scale_feats = []
        for i, block_size in enumerate(self.block_sizes):
            mag, phase = self._block_fft(x_f32, block_size)
            mag_feat   = self.mag_projs[i](mag.to(orig_dtype))
            phase_feat = self.phase_projs[i](phase.to(orig_dtype))
            scale_feats.append(mag_feat)
            scale_feats.append(phase_feat)

        fused = self.fuse(torch.cat(scale_feats, dim=1))
        return fused


class FrequencyBranch(nn.Module):

    def __init__(self, in_channels: int = 6, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 64,  3, stride=2, padding=1),   
            nn.GELU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),            
            nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),   
            nn.GELU(),
        )

        self.multiscale_fft = MultiScaleFFT(out_channels, block_sizes=(2, 4, 7))

        # ------------------------------------------------------------------
        # Post-FFT feature refinement
        # ------------------------------------------------------------------
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Channel attention: selects which frequency bands are most informative
        # Uses both avg and max pooling (richer statistics than avg alone)
        # ------------------------------------------------------------------
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 8),
            nn.GELU(),
            nn.Linear(out_channels // 8, out_channels),
            nn.Sigmoid(),
        )
        self.channel_attn_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 8),
            nn.GELU(),
            nn.Linear(out_channels // 8, out_channels),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Spatial attention: where in the feature map are frequency anomalies?
        # ------------------------------------------------------------------
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Final projection (ensures output residual has matching channels)
        # ------------------------------------------------------------------
        self.out_proj = nn.Conv2d(out_channels, out_channels, 1)

    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[2, 3], keepdim=True)
        std  = x.std(dim=[2, 3], keepdim=True) + 1e-6
        x    = (x - mean) / std
        return torch.clamp(x, -3.0, 3.0)

    def _channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        avg_attn = self.channel_attn(x).unsqueeze(-1).unsqueeze(-1)
        max_attn = self.channel_attn_max(x).unsqueeze(-1).unsqueeze(-1)
        return (avg_attn + max_attn) / 2.0

    def _spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        avg_proj = x.mean(dim=1, keepdim=True)
        max_proj = x.max(dim=1, keepdim=True)[0]
        return self.spatial_attn(torch.cat([avg_proj, max_proj], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat = self.adapter(x)     

        x_norm = self._normalize_features(x_feat)

        freq_feat = self.multiscale_fft(x_norm)     

        freq_feat = self.refine(freq_feat)

        ch_attn   = self._channel_attention(freq_feat)
        freq_feat = freq_feat * ch_attn

        sp_attn   = self._spatial_attention(freq_feat)
        freq_feat = freq_feat * sp_attn

        out = self.out_proj(freq_feat) + x_feat

        return out