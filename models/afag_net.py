import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import AFAGBackbone
from models.multi_branch_raw import MultiBranchModule      # spatial + noise unchanged
from models.branches.frequency_branch import FrequencyBranch  # updated
from models.fusion.cgaf import CGAF
from models.temporal.temporal_model import TemporalModel   # unchanged


# ==========================================================================
# IMPROVED OUTPUT HEADS
# ==========================================================================

class ClassificationHead(nn.Module):
    """
    Accepts concatenated [temporal_feat | high_feat_gated] → single logit.
    
    temporal_feat: (B, 512) from TemporalModel
    high_feat:     (B, 256) from F_high global pool
    combined:      (B, 768)
    """
    def __init__(self, temporal_dim: int = 512, high_dim: int = 256):
        super().__init__()
        combined_dim = temporal_dim + high_dim

        # Gate that learns how much F_high should influence classification
        self.high_gate = nn.Sequential(
            nn.Linear(high_dim, high_dim // 4),
            nn.GELU(),
            nn.Linear(high_dim // 4, high_dim),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            # Raw logit — no sigmoid. Apply sigmoid externally for inference.
            # Use with BCEWithLogitsLoss or FocalLoss during training.
        )

    def forward(self, temporal_feat: torch.Tensor, high_feat: torch.Tensor):
        gated_high = high_feat * self.high_gate(high_feat)
        combined   = torch.cat([temporal_feat, gated_high], dim=1)   # (B, 768)
        return self.fc(combined)


class LocalizationHead(nn.Module):
    """
    Decoder with F_high skip connection at 7x7 spatial level.
    
    Input:  Ffusion (B*N, 256, 14, 14) — main features
            F_high  (B*N, 256, 7, 7)  — global context for skip
    Output: (B*N, 1, 224, 224) — probability mask in [0, 1]
    
    Architecture: 14→28→56→112→224 via ConvTranspose2d
    F_high is injected at the 14×14 → 28×28 step
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()

        # Project F_high for skip connection
        self.high_skip_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.GELU(),
        )

        # Initial convolution on Ffusion
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
        )

        # Upsample with F_high skip at first step
        # After skip: channels = 256 + 128 = 384 → project back to 128
        self.skip_merge = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, 128, 1),
            nn.GELU(),
        )

        # Decoder: 14 → 28 → 56 → 112 → 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),   # 14→28
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),    # 28→56
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),    # 56→112
            nn.GELU(),
            nn.ConvTranspose2d(16,  8, 2, stride=2),    # 112→224
            nn.GELU(),
            nn.Conv2d(8, 1, 1),                         # channel squeeze
        )

    def forward(self, Ffusion: torch.Tensor, F_high: torch.Tensor = None):
        """
        Ffusion: (B*N, 256, 14, 14)
        F_high:  (B*N, 256, 7, 7)   — optional
        """
        x = self.init_conv(Ffusion)   # (B*N, 256, 14, 14)

        # Skip connection from F_high
        if F_high is not None:
            # Upsample F_high from 7→14
            skip = F.interpolate(F_high, size=(14, 14), mode='bilinear', align_corners=False)
            skip = self.high_skip_proj(skip)              # (B*N, 128, 14, 14)
            x    = self.skip_merge(torch.cat([x, skip], dim=1))  # (B*N, 128, 14, 14)
        else:
            # Fallback: project without skip
            x = self.skip_merge(torch.cat([x, torch.zeros_like(x[:, :128])], dim=1))

        return torch.sigmoid(self.decoder(x))  # (B*N, 1, 224, 224)


def confidence_map(mask):
    return torch.abs(mask - 0.5) * 2

def temporal_stability(mask_seq):
    diff = torch.abs(mask_seq[:, 1:] - mask_seq[:, :-1])
    return 1 - diff

def domain_attribution(Fs, Ff, Fn, Wf=None, Wn=None):
    freq  = Wf * Ff if Wf is not None else Ff
    noise = Wn * Fn if Wn is not None else Fn
    return {"spatial": Fs, "frequency": freq, "noise": noise}


class OutputHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_head = ClassificationHead()   # 768-dim input
        self.loc_head = LocalizationHead()     # with F_high skip

    def forward(
        self,
        video_feat:  torch.Tensor,    # (B, 512) temporal
        Ffusion:     torch.Tensor,    # (B, N, 256, 14, 14) fused
        F_high_flat: torch.Tensor,    # (B*N, 256, 7, 7)
        Fs=None, Ff=None, Fn=None,
    ):
        B, N, C, H, W = Ffusion.shape

        # --- Classification ---
        # F_high: aggregate across frames with mean pooling
        F_high_pool = F.adaptive_avg_pool2d(F_high_flat, 1)           # (B*N, 256, 1, 1)
        F_high_pool = F_high_pool.view(B, N, 256).mean(dim=1)          # (B, 256)
        pred_cls    = self.cls_head(video_feat, F_high_pool)            # (B, 1)

        # --- Localization ---
        F_flat     = Ffusion.view(B * N, C, H, W)
        masks_flat = self.loc_head(F_flat, F_high_flat)                 # (B*N, 1, 224, 224)
        mask_seq   = masks_flat.view(B, N, 1, 224, 224)
        pred_mask  = mask_seq[:, N // 2]                               # middle frame mask

        conf_map  = confidence_map(pred_mask)
        stability = temporal_stability(mask_seq)
        attribution = domain_attribution(Fs, Ff, Fn) if Fs is not None else None

        return {
            "pred_cls":      pred_cls,
            "pred_mask":     pred_mask,
            "mask_sequence": mask_seq,
            "confidence":    conf_map,
            "stability":     stability,
            "attribution":   attribution,
        }


# ==========================================================================
# AFAG NET  — Main model
# ==========================================================================

class AFAGNet(nn.Module):
    """
    AFAGNet  — all improvements integrated.
    
    Key differences from v1:
    - backbone_: ImageNet normalization fix + F_high projection
    - frequency_branch_raw: NaN-safe FFT + phase features
    - cgaf: bounded temperature + F_high gating
    - F_high flows through entire pipeline to output heads
    - ClassificationHead: 768-dim (temporal + global)
    - LocalizationHead: F_high skip for finer boundaries
    
    Usage is identical to v1 AFAGNet:
        model = AFAGNet()
        out   = model(x)   # x: (B, N, 6, 224, 224)
    """

    def __init__(self):
        super().__init__()

        self.backbone   = AFAGBackbone()       # backbone.py
        self.spatial    = __import__(
            'models.branches.spatial_branch', fromlist=['SpatialBranch']
        ).SpatialBranch()                         # unchanged from v1
        self.frequency  = FrequencyBranch()      # frequency_branch_raw.py
        self.noise      = __import__(
            'models.branches.noise_branch_raw', fromlist=['NoiseBranch']
        ).NoiseBranch()                           # unchanged from v1
        self.cgaf       = CGAF(use_f_high=True)  # cgaf.py
        self.temporal   = TemporalModel()         # unchanged from v1
        self.heads      = OutputHeads()

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, 6, 224, 224)
        """
        B, N, C, H, W = x.shape

        # Merge batch + temporal for frame-level processing
        x_flat = x.view(B * N, C, H, W)

        # -------- STEP 1: Backbone (now with F_high) --------
        F_low, F_high = self.backbone(x_flat)   # F_low: (B*N, 256, 14, 14)
                                                 # F_high: (B*N, 256, 7, 7)

        # -------- STEP 2: Multi-Branch --------
        Fs = self.spatial(F_low)                  # unchanged SpatialBranch
        Ff = self.frequency(x_flat)               # v2 FrequencyBranch (NaN-safe + phase)
        Fn = self.noise(x_flat)                   # unchanged NoiseBranch

        # -------- STEP 3: CGAF Fusion (now with F_high context) --------
        Ffusion_flat, Csf, Csn, Cfn = self.cgaf(Fs, Ff, Fn, F_high=F_high)

        # -------- STEP 4: Restore temporal dimension --------
        Ffusion = Ffusion_flat.view(B, N, 256, 14, 14)

        # -------- STEP 5: Temporal Modeling --------
        video_feat = self.temporal(Ffusion)   # (B, 512)

        # -------- STEP 6: Output Heads --------
        head_out = self.heads(
            video_feat,
            Ffusion,
            F_high_flat=F_high,              # NEW: pass F_high for skip connection
            Fs=Fs, Ff=Ff, Fn=Fn,
        )

        return {
            **head_out,
            "F_low":        F_low,
            "F_high":       F_high,
            "Fs":           Fs,
            "Ff":           Ff,
            "Fn":           Fn,
            "Ffusion":      Ffusion_flat,
            "Ffusion_seq":  Ffusion,
            "Csf":          Csf,
            "Csn":          Csn,
            "Cfn":          Cfn,
        }


# ==========================================================================
# AltFreezing training helper
# Reference: AltFreezing (ECCV 2023)
# ==========================================================================

class AltFreezingTrainer:
    """
    Alternate between freezing backbone+branches vs freezing temporal+heads.
    
    Phase A (odd epochs): freeze backbone+branches, train temporal+heads
    Phase B (even epochs): freeze temporal+heads, train backbone+branches
    
    Benefits:
    - Prevents gradient interference between spatial and temporal components
    - Reduces peak VRAM (frozen modules have no gradient storage)
    - Leads to better temporal feature quality (consistent spatial input)
    
    Usage:
        alt_trainer = AltFreezingTrainer(model)
        for epoch in range(epochs):
            alt_trainer.set_epoch(epoch)
            # ... train normally
    """

    def __init__(self, model: AFAGNet):
        self.model = model

        self.spatial_modules  = [model.backbone, model.spatial,
                                 model.frequency, model.noise, model.cgaf]
        self.temporal_modules = [model.temporal, model.heads]

    def _set_trainable(self, modules, trainable: bool):
        for m in modules:
            for p in m.parameters():
                p.requires_grad = trainable

    def set_epoch(self, epoch: int):
        if epoch % 2 == 0:
            # Phase A: train temporal, freeze spatial
            self._set_trainable(self.spatial_modules,  trainable=False)
            self._set_trainable(self.temporal_modules, trainable=True)
            mode = "temporal"
        else:
            # Phase B: train spatial, freeze temporal
            self._set_trainable(self.spatial_modules,  trainable=True)
            self._set_trainable(self.temporal_modules, trainable=False)
            mode = "spatial+branches"
        print(f"AltFreezing — Epoch {epoch+1}: training {mode}")

