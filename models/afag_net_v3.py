
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import AFAGBackbone
from models.branches.spatial_branch import SpatialBranch
from models.branches.frequency_branch_raw import FrequencyBranch   
from models.branches.noise_branch_raw import NoiseBranch
from models.fusion.cgaf import CGAFv2                  
from models.temporal.temporal_model import TemporalModel


# ===========================================================================
# CLASSIFICATION HEAD 
# ===========================================================================

class ClassificationHead(nn.Module):
    def __init__(self, temporal_dim: int = 512, high_dim: int = 256):
        super().__init__()

        self.high_gate = nn.Sequential(
            nn.Linear(high_dim, high_dim // 4),
            nn.GELU(),
            nn.Linear(high_dim // 4, high_dim),
            nn.Sigmoid(),
        )

        combined_dim = temporal_dim + high_dim
        self.fc = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(0.3),          
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),          
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),          
            nn.Linear(64, 1),
        )

    def forward(self, temporal_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        gated    = high_feat * self.high_gate(high_feat)
        combined = torch.cat([temporal_feat, gated], dim=1)
        return self.fc(combined)


# ===========================================================================
# LOCALIZATION HEAD 
# ===========================================================================

class LocalizationHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        self.high_skip_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.GELU(),
        )

        self.skip_merge = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, 128, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.no_skip_proj = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),   
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),     
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),    
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, Ffusion: torch.Tensor, F_high: torch.Tensor = None) -> torch.Tensor:
        x = self.init_conv(Ffusion)   

        if F_high is not None:
            skip = F.interpolate(F_high, size=Ffusion.shape[2:],
                                 mode='bilinear', align_corners=False)
            skip = self.high_skip_proj(skip)                       
            x    = self.skip_merge(torch.cat([x, skip], dim=1))    
        else:
            x = self.no_skip_proj(x)

        return torch.sigmoid(self.decoder(x))  


# ===========================================================================
# OUTPUT HEADS 
# ===========================================================================

class OutputHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_head = ClassificationHead()
        self.loc_head = LocalizationHead()

    def forward(
        self,
        video_feat:  torch.Tensor,   
        Ffusion:     torch.Tensor,   
        F_high_flat: torch.Tensor,   
        Fs=None, Ff=None, Fn=None,
    ):
        B, N, C, H, W = Ffusion.shape

        F_high_pool = F.adaptive_avg_pool2d(F_high_flat, 1).view(B, N, 256).mean(dim=1)
        pred_cls    = self.cls_head(video_feat, F_high_pool) 

        F_flat     = Ffusion.view(B * N, C, H, W)
        masks_flat = self.loc_head(F_flat, F_high_flat)        
        mask_seq   = masks_flat.view(B, N, 1, 224, 224)
        pred_mask  = mask_seq[:, N // 2]                      

        conf_map  = torch.abs(pred_mask - 0.5) * 2
        stability = 1 - torch.abs(mask_seq[:, 1:] - mask_seq[:, :-1])

        return {
            "pred_cls":      pred_cls,
            "pred_mask":     pred_mask,
            "mask_sequence": mask_seq,
            "confidence":    conf_map,
            "stability":     stability,
        }


# ===========================================================================
# MAIN MODEL
# ===========================================================================

class AFAGNetV3(nn.Module):

    def __init__(self, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.backbone   = AFAGBackbone()
        self.spatial    = SpatialBranch(channels=256, high_channels=256)
        self.frequency  = FrequencyBranch()
        self.noise      = NoiseBranch()
        self.cgaf       = CGAFv2(channels=256, use_f_high=True)  
        self.temporal   = TemporalModel()
        self.heads      = OutputHeads()

    def forward(self, x: torch.Tensor) -> dict:
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)

        # Backbone 
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            F_low, F_high = checkpoint(self.backbone, x_flat)     
        else:
            F_low, F_high = self.backbone(x_flat)

        Fs = self.spatial(F_low, F_high)    
        Ff = self.frequency(x_flat)    
        Fn = self.noise(x_flat)             

        target_dtype = F_low.dtype
        Fs = Fs.to(target_dtype)
        Ff = Ff.to(target_dtype)
        Fn = Fn.to(target_dtype)

        # CGAFv2 Fusion
        Ffusion_flat, Csf, Csn, Cfn = self.cgaf(Fs, Ff, Fn, F_high=F_high)
        # Ffusion_flat: (B*N, 256, 28, 28)

        # Restore temporal dimension
        _, C_f, H_f, W_f = Ffusion_flat.shape
        Ffusion = Ffusion_flat.view(B, N, C_f, H_f, W_f)

        # Temporal modeling 
        video_feat = self.temporal(Ffusion)   

        # Output heads
        head_out = self.heads(
            video_feat,
            Ffusion,
            F_high_flat=F_high,
            Fs=Fs, Ff=Ff, Fn=Fn,
        )

        return {
            **head_out,
            "F_low":       F_low,
            "F_high":      F_high,
            "Fs":          Fs,
            "Ff":          Ff,
            "Fn":          Fn,
            "Ffusion":     Ffusion_flat,
            "Ffusion_seq": Ffusion,
            "Csf":         Csf,
            "Csn":         Csn,
            "Cfn":         Cfn,
        }


# ===========================================================================
# AltFreezing Trainer 
# ===========================================================================

class AltFreezingTrainer:
    """
    Alternate training phases:
      Even epochs: freeze spatial modules, train temporal+heads
      Odd  epochs: freeze temporal+heads, train spatial modules

    Benefits:
     Prevents gradient interference between spatial and temporal paths
     Reduces VRAM in frozen phase (no gradient storage for frozen params)
     Leads to better temporal discrimination 

    Usage:
        alt = AltFreezingTrainer(model)
        for epoch in range(epochs):
            alt.set_epoch(epoch)
            # ... train normally
    """

    def __init__(self, model: AFAGNetV3):
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
            self._set_trainable(self.spatial_modules,  trainable=False)
            self._set_trainable(self.temporal_modules, trainable=True)
            mode = "temporal+heads"
        else:
            self._set_trainable(self.spatial_modules,  trainable=True)
            self._set_trainable(self.temporal_modules, trainable=False)
            mode = "spatial+branches+cgaf"
        print(f"[AltFreezing] Epoch {epoch+1}: training {mode}")
