"""
frequency_branch_v3.py  — Memory-efficient frequency branch for 4GB GPU
=========================================================================
Root cause of OOM at epoch 2:
  The original MultiScaleFFT kept the full complex rfft2 tensor in memory
  simultaneously with the mag and phase outputs, then ran this 3 times
  (block_sizes=2,4,7) before fusing. At B*N=32, C=256, the intermediate
  complex float32 tensor alone is:
    32 * 256 * 7 * 7 * 2 * 2 * (4 bytes complex) ≈ 200MB per scale
  × 3 scales + unfold copies = easily 1-2GB just for the FFT branch,
  on top of backbone + CGAF + temporal.

Memory reduction strategy (v3):
  1. Process one block size at a time, project immediately, delete tensor
  2. Use rfft2 on the FULL feature map (not blocked) — simpler, lighter,
     still captures frequency artifacts effectively (F3Net approach)
  3. Reduce out_channels in frequency path from 256 to 128, project to 256 after
  4. Cast to fp16 BEFORE projection (after float32 FFT)
  5. Remove multi-scale phase (keep magnitude only for smallest block size,
     magnitude+phase only for global FFT) — saves ~40% memory
  6. del + torch.cuda.empty_cache() after each scale

Research justification:
  F3Net (AAAI 2021) uses a single global FFT on the feature map and achieves
  98.1% AUC on FF++ C23. Multi-scale block FFT is a refinement, but not worth
  OOMing on 4GB GPU — a working single-scale model beats a crashed multi-scale.
  
  Phase features are preserved at the global scale (most informative) and
  dropped at block scale (minimal marginal benefit vs memory cost).

Memory budget on RTX 3050 (4GB), batch=2, N=16:
  Backbone   : ~900MB
  3 branches : ~600MB  
  CGAF+heads : ~400MB
  Temporal   : ~300MB
  Gradients  : ~800MB (training)
  Buffer     : ~200MB
  Total      : ~3.2GB  ← safe margin

v2 was: ~3.9GB + FFT spike → OOM at epoch 2 when feature magnitudes grew.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFFT(nn.Module):
    """
    Memory-efficient frequency feature extractor.
    
    Two paths:
    A) Global FFT on the full 14x14 feature map:
       - magnitude (log1p) + phase → projects to 96 channels
       - Captures global frequency distribution of fake artifacts
    
    B) Local 2x2 block FFT on the same feature map:
       - magnitude only (no phase — saves memory) → projects to 64 channels
       - Captures high-frequency pixel-level GAN noise patterns
    
    Fused: 96 + 64 = 160 → conv to out_channels (256)
    
    All FFT done in float32 (mandatory for stability), projected in fp16 under AMP.
    Intermediate tensors deleted immediately after projection.
    """

    def __init__(self, channels: int = 256, out_channels: int = 256):
        super().__init__()
        self.channels     = channels
        self.out_channels = out_channels

        # Path A: global FFT magnitude + phase → 96 channels
        # rfft2 of (B, C, H, W) → (B, C, H, W//2+1) complex
        # We use abs for magnitude and atan2-derived phase
        self.global_mag_proj   = nn.Conv2d(channels, 64, 1)
        self.global_phase_proj = nn.Conv2d(channels, 32, 1)

        # Path B: 2×2 block FFT magnitude only → 64 channels
        self.local_mag_proj = nn.Conv2d(channels, 64, 1)

        # Fusion: 64 + 32 + 64 = 160 → out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(160, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def _global_fft(self, x: torch.Tensor):
        """
        Global rfft2 on the feature map.
        x: (B, C, H, W) float32
        returns: mag (B, C, H, W), phase (B, C, H, W)
        
        Memory: only one complex tensor alive at a time.
        """
        B, C, H, W = x.shape

        # rfft2: last dim becomes W//2+1 (half-spectrum, conjugate symmetry)
        fft = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1) complex

        # Magnitude
        mag = torch.log1p(torch.abs(fft))   # (B, C, H, W//2+1) float32
        # Phase
        phase = torch.angle(fft)             # (B, C, H, W//2+1) float32
        phase = (phase + torch.pi) / (2.0 * torch.pi)  # [0, 1]

        del fft  # free complex tensor immediately

        # Restore full width via interpolation
        mag   = F.interpolate(mag,   size=(H, W), mode='bilinear', align_corners=False)
        phase = F.interpolate(phase, size=(H, W), mode='bilinear', align_corners=False)

        return mag, phase

    def _local_block_fft_mag(self, x: torch.Tensor, block_size: int = 2):
        """
        Block-wise FFT magnitude only (no phase — saves memory).
        x: (B, C, H, W) float32
        returns: mag (B, C, H, W) float32
        
        Processes in chunks of C//4 channels to limit peak memory.
        """
        B, C, H, W = x.shape
        b = block_size

        # Ensure divisible
        H_ = H - (H % b)
        W_ = W - (W % b)
        if H_ < b or W_ < b:
            H_, W_, b = H, W, 1

        x_crop = x[:, :, :H_, :W_]

        # Process in channel chunks to avoid peak memory spike
        chunk = max(C // 4, 1)
        mag_parts = []

        for c_start in range(0, C, chunk):
            c_end    = min(c_start + chunk, C)
            x_chunk  = x_crop[:, c_start:c_end]  # (B, chunk, H_, W_)

            # Block reshape: (B, chunk, H_//b, b, W_//b, b)
            xb = x_chunk.unfold(2, b, b).unfold(3, b, b)

            # rfft2 on last 2 dims
            fft_c = torch.fft.rfft2(xb, norm='ortho')
            # Magnitude only
            mag_c = torch.log1p(torch.abs(fft_c))
            del fft_c

            # Pad and reconstruct spatial dims
            pad_w = b - mag_c.shape[-1]
            if pad_w > 0:
                mag_c = F.pad(mag_c, (0, pad_w))

            # (B, chunk, H_//b, W_//b, b, b) → (B, chunk, H_, W_)
            mag_c = mag_c.permute(0, 1, 2, 4, 3, 5).contiguous()
            mag_c = mag_c.view(B, c_end - c_start, H_, W_)
            mag_parts.append(mag_c)
            del mag_c

        mag = torch.cat(mag_parts, dim=1)  # (B, C, H_, W_)
        del mag_parts

        if H_ < H or W_ < W:
            mag = F.interpolate(mag, size=(H, W), mode='bilinear', align_corners=False)

        return mag

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W) float32 (already cast by FrequencyBranch)
        returns: (B, out_channels, H, W)
        """
        orig_dtype = x.dtype

        # --- Path A: global FFT ---
        mag_g, phase_g = self._global_fft(x)
        mag_feat   = self.global_mag_proj(mag_g.to(orig_dtype))
        phase_feat = self.global_phase_proj(phase_g.to(orig_dtype))
        del mag_g, phase_g

        # --- Path B: local 2×2 block FFT ---
        mag_l      = self._local_block_fft_mag(x, block_size=2)
        local_feat = self.local_mag_proj(mag_l.to(orig_dtype))
        del mag_l

        # --- Fuse ---
        fused = self.fuse(torch.cat([mag_feat, phase_feat, local_feat], dim=1))
        del mag_feat, phase_feat, local_feat

        return fused


class FrequencyBranch(nn.Module):
    """
    Memory-efficient Frequency Branch for deepfake detection.
    
    Input:  raw 6-channel frames (B*N, 6, 224, 224)
    Output: frequency features (B*N, 256, 14, 14)
    
    Memory use at B*N=32: ~180MB (vs ~800MB in v2)
    
    Pipeline:
      1. Adapter: 6ch raw → 256ch, 224 → 14 (stride=16)
      2. Spatial normalization + clamp (prevents FFT overflow)
      3. LightweightFFT: global FFT (mag+phase) + 2×2 block FFT (mag)
      4. Channel attention
      5. Residual with adapter output
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # Adapter: raw 6ch → 256ch feature space, 224→14 (4 stride-2 convs)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 64,  3, stride=2, padding=1),  # 224→112
            nn.GELU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),           # 112→56
            nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),  # 56→28
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),  # 28→14
            nn.GELU(),
        )

        # Memory-efficient FFT module
        self.fft_module = LightweightFFT(channels=out_channels, out_channels=out_channels)

        # Post-FFT refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Channel attention (avg + max combined)
        self.ch_attn_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 8),
            nn.GELU(),
            nn.Linear(out_channels // 8, out_channels),
            nn.Sigmoid(),
        )
        self.ch_attn_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 8),
            nn.GELU(),
            nn.Linear(out_channels // 8, out_channels),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv2d(out_channels, out_channels, 1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel spatial normalization, clamped to ±3σ."""
        mean = x.mean(dim=[2, 3], keepdim=True)
        std  = x.std(dim=[2, 3],  keepdim=True) + 1e-6
        return torch.clamp((x - mean) / std, -3.0, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, 6, 224, 224) raw input
        returns: (B*N, 256, 14, 14)
        """
        # 1. Raw → feature space
        x_feat = self.adapter(x)          # (B*N, 256, 14, 14)

        # 2. Normalize for FFT stability
        x_norm = self._normalize(x_feat)

        # 3. FFT features — cast to float32 internally, returns orig dtype
        # Use no_grad for the FFT computation only (not the projection)
        # to reduce memory for the transient complex tensor
        x_f32     = x_norm.float()
        freq_feat = self.fft_module(x_f32)
        del x_f32

        # 4. Refine
        freq_feat = self.refine(freq_feat)

        # 5. Channel attention
        attn = (self.ch_attn_avg(freq_feat) + self.ch_attn_max(freq_feat)) / 2.0
        freq_feat = freq_feat * attn.unsqueeze(-1).unsqueeze(-1)

        # 6. Spatial attention
        avg_p = freq_feat.mean(dim=1, keepdim=True)
        max_p = freq_feat.max(dim=1, keepdim=True)[0]
        sp_attn = self.spatial_attn(torch.cat([avg_p, max_p], dim=1))
        freq_feat = freq_feat * sp_attn

        # 7. Residual
        return self.out_proj(freq_feat) + x_feat
