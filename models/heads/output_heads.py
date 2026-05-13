import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# CLASSIFICATION HEAD
# ---------------------------
class ClassificationHead(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Returns RAW LOGITS — no sigmoid here.
        # During training: BCEWithLogitsLoss applies sigmoid internally.
        # During eval/inference: apply torch.sigmoid() manually before threshold.
        # This is numerically more stable than sigmoid → BCELoss.
        return self.fc(x)


# ---------------------------
# LOCALIZATION HEAD
# ---------------------------
class LocalizationHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.decoder(x))


# ---------------------------
# EXPLAINABILITY
# ---------------------------
def confidence_map(mask):
    return torch.abs(mask - 0.5) * 2


def temporal_stability(mask_seq):
    diff = torch.abs(mask_seq[:, 1:] - mask_seq[:, :-1])
    return 1 - diff


def domain_attribution(Fs, Ff, Fn, Wf=None, Wn=None):
    freq  = Wf * Ff if Wf is not None else Ff
    noise = Wn * Fn if Wn is not None else Fn
    return {"spatial": Fs, "frequency": freq, "noise": noise}


# ---------------------------
# FULL OUTPUT MODULE
# ---------------------------
class OutputHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_head = ClassificationHead()
        self.loc_head = LocalizationHead()

    def forward(self, video_feat, Ffusion, Fs=None, Ff=None, Fn=None, Wf=None, Wn=None):
        pred_cls = self.cls_head(video_feat)  # raw logits

        B, N, C, H, W = Ffusion.shape
        F_flat     = Ffusion.view(B * N, C, H, W)
        masks_flat = self.loc_head(F_flat)
        mask_seq   = masks_flat.view(B, N, 1, masks_flat.shape[-2], masks_flat.shape[-1])
        pred_mask  = mask_seq[:, N // 2]

        conf_map  = confidence_map(pred_mask)
        stability = temporal_stability(mask_seq)
        attribution = domain_attribution(Fs, Ff, Fn, Wf, Wn) if Fs is not None else None

        return {
            "pred_cls":      pred_cls,
            "pred_mask":     pred_mask,
            "mask_sequence": mask_seq,
            "confidence":    conf_map,
            "stability":     stability,
            "attribution":   attribution,
        }
