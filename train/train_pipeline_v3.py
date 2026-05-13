import os
import time
import math
import copy
import warnings
from datetime import datetime
from collections import defaultdict

try:
    from train.psuedo_mask import generate_pseudo_mask
    _PSEUDO_MASK_AVAILABLE = True
except ImportError:
    try:
        from psuedo_mask import generate_pseudo_mask
        _PSEUDO_MASK_AVAILABLE = True
    except ImportError:
        _PSEUDO_MASK_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*flash attention.*",     category=UserWarning)
warnings.filterwarnings("ignore", message=".*ComplexHalf.*",         category=UserWarning)
warnings.filterwarnings("ignore", message=".*rcond parameter.*",     category=FutureWarning)
warnings.filterwarnings("ignore", message=".*HF Hub.*",              category=UserWarning)
warnings.filterwarnings("ignore", message=".*Only one class.*",      category=UserWarning)
warnings.filterwarnings("ignore", message=".*UndefinedMetricWarning.*")


# ===========================================================================
# TRAINING GRAPH GENERATOR
# Saves after training completes:
#   training_history.png  — 4-panel: Loss / AUC / Accuracy breakdown / LR schedule
#   overfit_monitor.png   — Train-Val AUC gap per epoch (overfitting detector)
# ===========================================================================
def plot_training_history(history: dict, save_dir: str, batch_size: int = 2):
    os.makedirs(save_dir, exist_ok=True)

    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    def clean(lst):
        return [v if (v is not None and not (isinstance(v, float) and math.isnan(v)))
                else float("nan") for v in lst]

    style = {
        "train": dict(color="#2563EB", linewidth=2.0, marker="o", markersize=4),
        "val":   dict(color="#DC2626", linewidth=2.0, marker="s", markersize=4),
        "real":  dict(color="#16A34A", linewidth=1.5, linestyle="--"),
        "fake":  dict(color="#D97706", linewidth=1.5, linestyle="--"),
        "bb":    dict(color="#7C3AED", linewidth=1.5),
        "rest":  dict(color="#0891B2", linewidth=1.5),
    }

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"AFAGNetV3 Training History  [batch={batch_size}]",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # ── Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if history.get("train_loss"):
        ax1.plot(epochs, clean(history["train_loss"]), label="Train Loss", **style["train"])
    if history.get("val_loss"):
        ax1.plot(epochs, clean(history["val_loss"]),   label="Val Loss",   **style["val"])
    ax1.set_title("Loss vs Epoch", fontweight="bold"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    # ── AUC
    ax2 = fig.add_subplot(gs[0, 1])
    if history.get("train_auc"):
        ax2.plot(epochs, clean(history["train_auc"]), label="Train AUC", **style["train"])
    if history.get("val_auc"):
        ax2.plot(epochs, clean(history["val_auc"]),   label="Val AUC",   **style["val"])
    val_auc_clean = [v for v in clean(history.get("val_auc", [])) if not math.isnan(v)]
    if val_auc_clean:
        best_v = max(val_auc_clean)
        best_i = val_auc_clean.index(best_v)
        ax2.annotate(f"Best: {best_v:.4f}",
                     xy=(epochs[best_i], best_v),
                     xytext=(epochs[best_i], best_v - 0.04),
                     fontsize=9, ha="center", color="#DC2626",
                     arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2))
    ax2.set_title("AUC vs Epoch", fontweight="bold"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC"); ax2.legend(); ax2.grid(alpha=0.3)

    # ── Accuracy breakdown
    ax3 = fig.add_subplot(gs[1, 0])
    if history.get("train_acc"):
        ax3.plot(epochs, [v*100 for v in clean(history["train_acc"])],
                 label="Train Overall", **style["train"])
    if history.get("val_acc"):
        ax3.plot(epochs, [v*100 for v in clean(history["val_acc"])],
                 label="Val Overall", **style["val"])
    if history.get("train_real_acc"):
        ax3.plot(epochs, [v*100 for v in clean(history["train_real_acc"])],
                 label="Train Real", **style["real"])
    if history.get("train_fake_acc"):
        ax3.plot(epochs, [v*100 for v in clean(history["train_fake_acc"])],
                 label="Train Fake", **style["fake"])
    ax3.set_title("Accuracy vs Epoch", fontweight="bold"); ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # ── LR schedule
    ax4 = fig.add_subplot(gs[1, 1])
    if history.get("lr_backbone"):
        ax4.semilogy(epochs, history["lr_backbone"], label="Backbone LR", **style["bb"])
    if history.get("lr_rest"):
        ax4.semilogy(epochs, history["lr_rest"],    label="Rest LR",    **style["rest"])
    ax4.set_title("Learning Rate Schedule", fontweight="bold"); ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR (log scale)"); ax4.legend(); ax4.grid(alpha=0.3, which="both")

    plt.savefig(os.path.join(save_dir, "training_history.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Overfit monitor
    if history.get("train_auc") and history.get("val_auc"):
        ta = clean(history["train_auc"]); va = clean(history["val_auc"])
        gap = [t-v if not (math.isnan(t) or math.isnan(v)) else float("nan")
               for t, v in zip(ta, va)]
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.bar(epochs, gap,
               color=["#DC2626" if (not math.isnan(g) and g>0.05) else "#16A34A"
                      for g in gap], alpha=0.7)
        ax.axhline(y=0.05, color="orange", linestyle="--", label="Overfit threshold (0.05)")
        ax.axhline(y=0.0,  color="gray",   linestyle="-",  linewidth=0.8)
        ax.set_title("Train−Val AUC Gap (Overfitting Monitor)", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train AUC − Val AUC")
        ax.legend(); ax.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, "overfit_monitor.png"),
                     dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig2)

    print(f"  Graphs saved → {save_dir}/")


# ===========================================================================
# FOCAL LOSS
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss with class-frequency-aware alpha.

    FF++ C23 class ratio: ~1082 real / 4979 fake = 18% real, 82% fake.
    alpha=0.5 biases the model toward predicting FAKE (larger class dominates).
    This is why the optimal threshold was 0.10 — the model over-predicts fake.

    Correct alpha for balanced gradient contribution:
      alpha = 0.25 → strongly upweights real (minority) class gradients.

    Mathematical form:
      loss = -α_t (1-p_t)^γ log(p_t)
      where α_t = alpha for positive (fake) class, (1-alpha) for negative (real)
      With alpha=0.25: real gets weight 0.75, fake gets 0.25
      This compensates for the 4.6:1 fake:real imbalance.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss   = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


# ===========================================================================
# FIX-3: CORRECT CONSISTENCY LOSS
# ===========================================================================
def branch_consistency_loss(Fs, Ff, Fn):
    """
    Cosine similarity consistency between branches.
    NaN-safe: adds eps to norms before division, clamps output to valid range.
    """
    fs_p = F.adaptive_avg_pool2d(Fs, 1).flatten(1)   # (B*N, 256)
    ff_p = F.adaptive_avg_pool2d(Ff, 1).flatten(1)
    fn_p = F.adaptive_avg_pool2d(Fn, 1).flatten(1)

    # NaN-safe cosine similarity: normalize manually with eps guard
    fs_n = F.normalize(fs_p, dim=1, eps=1e-8)
    ff_n = F.normalize(ff_p, dim=1, eps=1e-8)
    fn_n = F.normalize(fn_p, dim=1, eps=1e-8)

    cos_sf = (fs_n * ff_n).sum(dim=1).clamp(-1.0, 1.0)
    cos_sn = (fs_n * fn_n).sum(dim=1).clamp(-1.0, 1.0)

    loss = (1.0 - cos_sf).mean() + (1.0 - cos_sn).mean()

    # Safety: return 0 if NaN (e.g. near-zero features during LR restart)
    if torch.isnan(loss):
        return torch.tensor(0.0, device=Fs.device, requires_grad=False)
    return loss


def localization_loss(pred_mask, gt_mask, has_mask, pseudo_mask=None):
    real_loss   = torch.tensor(0.0, device=pred_mask.device)
    pseudo_loss = torch.tensor(0.0, device=pred_mask.device)
    if has_mask.sum() > 0:
        real_loss = F.l1_loss(pred_mask[has_mask], gt_mask[has_mask])
    if (~has_mask).sum() > 0 and pseudo_mask is not None:
        pseudo_loss = F.l1_loss(pred_mask[~has_mask], pseudo_mask[~has_mask])
    return real_loss + 0.3 * pseudo_loss


def temporal_consistency_loss(mask_seq):
    """Penalize large frame-to-frame mask changes. float32 cast prevents fp16 issues."""
    diff = torch.abs(mask_seq[:, 1:].float() - mask_seq[:, :-1].float())
    diff = torch.clamp(diff, 0.0, 10.0)   # cap at 10 — prevents rare extreme values
    return diff.mean()


# ===========================================================================
# NEW-1: MixUp AUGMENTATION
# ===========================================================================

def mixup_batch(frames: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """
    Video-level MixUp augmentation.
    
    Randomly mixes two video clips with coefficient λ ~ Beta(alpha, alpha).
    Mixed label is a convex combination of the two labels.
    
    Only applied 50% of the time (skip_mixup flag).
    
    Args:
        frames: (B, N, C, H, W)
        labels: (B, 1)
        alpha:  Beta distribution parameter (0.2 = mild mixing)
    
    Returns: (mixed_frames, labels_a, labels_b, lam)
    """
    if alpha <= 0.0:
        return frames, labels, labels, 1.0

    lam   = torch.distributions.Beta(alpha, alpha).sample().item()
    lam   = max(lam, 1.0 - lam)   # ensure lam >= 0.5 (dominant sample)

    B     = frames.size(0)
    index = torch.randperm(B, device=frames.device)

    mixed_frames = lam * frames + (1.0 - lam) * frames[index]
    labels_a     = labels
    labels_b     = labels[index]

    return mixed_frames, labels_a, labels_b, lam


def mixup_loss(criterion, logits, labels_a, labels_b, lam):
    """Loss for MixUp: λ * loss(a) + (1-λ) * loss(b)."""
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


# ===========================================================================
# NEW-5: EMA (Exponential Moving Average)
# ===========================================================================

class ModelEMA:
    """
    EMA of model weights for evaluation.
    EMA model is not trained, just maintained as a running average.
    
    Usage:
        ema = ModelEMA(model, decay=0.9999)
        # after each optimizer.step():
        ema.update(model)
        # for validation:
        ema.apply_to(eval_model)   # or use ema.model directly
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# ===========================================================================
# WARMUP SCHEDULER (unchanged from v2)
# ===========================================================================

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs: int, after_scheduler):
        self.optimizer       = optimizer
        self.warmup_epochs   = warmup_epochs
        self.after_scheduler = after_scheduler
        self.base_lrs        = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base_lr * scale
        else:
            # Don't pass epoch — deprecated in PyTorch 1.1+, causes warnings
            self.after_scheduler.step()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


# ===========================================================================
# FIX-1: FAST BALANCED SAMPLER
# ===========================================================================

def make_balanced_sampler(dataset):
    """
    FIX-1: O(N) label extraction from dataset.labels list (no disk access).
    Supports FFPPDatasetV2 directly and torch.utils.data.Subset.
    """
    if hasattr(dataset, "labels"):
        labels = [int(l) for l in dataset.labels]
    elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base   = dataset.dataset
        labels = [int(base.labels[i]) for i in dataset.indices]
    else:
        # Slow fallback — only if neither attribute exists
        print("WARNING: Using slow label extraction. Use FFPPDatasetV2 for fast sampler.")
        labels = [int(dataset[i]["label"].item()) for i in range(len(dataset))]

    n_real = labels.count(0)
    n_fake = labels.count(1)
    print(f"\nClass balance — real: {n_real}, fake: {n_fake}")

    w_real = 1.0 / (n_real + 1e-6)
    w_fake = 1.0 / (n_fake + 1e-6)
    weights = [w_real if l == 0 else w_fake for l in labels]

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ===========================================================================
# BATCH METRIC HELPERS
# ===========================================================================

def batch_per_class_accuracy(pred_logits, labels):
    probs     = torch.sigmoid(pred_logits)
    preds     = (probs >= 0.5).float()
    # Round labels to 0/1 before comparison — handles label smoothing (0.05/0.95)
    hard_labels = labels.round()
    correct   = (preds == hard_labels).float()
    real_mask = (hard_labels < 0.5).float()
    fake_mask = (hard_labels >= 0.5).float()
    overall   = correct.mean().item()
    real_acc  = (correct * real_mask).sum() / (real_mask.sum() + 1e-6)
    fake_acc  = (correct * fake_mask).sum() / (fake_mask.sum() + 1e-6)
    return overall, real_acc.item(), fake_acc.item()


# ===========================================================================
# SYSTEM INFO
# ===========================================================================

def print_system_info(device, model):
    print("\n" + "=" * 70)
    print("SYSTEM & MODEL INFORMATION")
    print("=" * 70)
    if device.type == "cuda":
        print(f"Device:     GPU (CUDA)")
        print(f"GPU Model:  {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory: {mem:.2f} GB")
        print(f"CUDA:       {torch.version.cuda}")
    else:
        print("Device:     CPU")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PyTorch:    {torch.__version__}")
    print(f"Params:     {total:,} total / {trainable:,} trainable")
    print("=" * 70 + "\n")


# ===========================================================================
# TRAINING STEP
# ===========================================================================

def train_one_epoch(
    model, loader, optimizer, device, scaler,
    epoch, total_epochs, use_amp, cls_loss_fn,
    warmup_epochs=2, use_mixup=True, mixup_alpha=0.2,
    ema=None,
):
    model.train()

    running_loss        = 0.0
    running_correct     = 0.0
    running_real_acc    = 0.0
    running_fake_acc    = 0.0
    total_samples       = 0.0
    nan_count           = 0
    all_probs           = []
    all_labels_flat     = []
    loc_weight = max(0.1, 0.8 - epoch * 0.12)
    # FIXED: previous max(0.3, 1.0 - epoch*0.15) decayed too slowly
    # and the 0.3 floor kept localization loss high enough to distort
    # classification logits toward mask-space, causing oscillating val AUC.
    # New schedule: 0.8→0.1 over 6 epochs, then stays at 0.1 (safe floor)

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [TRAIN]",
                    leave=True, dynamic_ncols=True)

    for batch_idx, batch in enumerate(progress):
        frames   = batch["frames"].to(device)
        labels   = batch["label"].to(device).unsqueeze(1)
        masks    = batch["mask_frames"].to(device)
        has_mask = batch["has_mask"].to(device)
        if has_mask.dtype != torch.bool:
            has_mask = has_mask.bool()

        # NEW-1: MixUp (50% chance during training, skip during warmup AND epoch 3)
        # Disabled for first 3 epochs — model needs to learn basic features first
        do_mixup = use_mixup and epoch >= max(warmup_epochs, 3) and torch.rand(1).item() < 0.5
        if do_mixup:
            frames, labels_a, labels_b, lam = mixup_batch(frames, labels, alpha=mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out      = model(frames)
            pred_cls  = out["pred_cls"]
            pred_mask = out["pred_mask"]

            Fs, Ff, Fn = out.get("Fs"), out.get("Ff"), out.get("Fn")

            # Classification loss
            with torch.cuda.amp.autocast(enabled=False):
                if do_mixup:
                    l_cls = mixup_loss(cls_loss_fn,
                                       pred_cls.float(), labels_a.float(),
                                       labels_b.float(), lam)
                else:
                    l_cls = cls_loss_fn(pred_cls.float(), labels.float())

            # Localization loss — real GT masks where available,
            # pseudo-mask (from freq+noise branches) for samples without GT masks
            mid = masks[:, masks.shape[1] // 2].unsqueeze(1)

            pseudo_mask = None
            if _PSEUDO_MASK_AVAILABLE and Fs is not None and Ff is not None and Fn is not None:
                try:
                    # use_gradcam=False → uses Ff + Fn only, no 2nd forward pass
                    # Safe on 4GB GPU — avoids storing gradients for entire model twice
                    #
                    # TO ENABLE GRADCAM ON T4 (16GB): change use_gradcam=True
                    # GradCAM adds ~1GB VRAM but gives better pseudo-mask quality
                    # (+0.3–0.5% localization IoU improvement).
                    # NEVER enable inside torch.no_grad() — needs autograd.
                    pseudo_mask = generate_pseudo_mask(
                        model, frames, Fs, Ff, Fn,
                        alpha=0.5, beta=0.3, gamma=0.2,
                        use_gradcam=True,   # ← change to True on 16GB GPU for better masks (requires more VRAM
                    )
                    # Resize pseudo_mask to (B,1,224,224) middle frame only
                    N_frames = frames.shape[1]
                    pseudo_mask = pseudo_mask[:, N_frames // 2]  # (B,1,H,W)
                except Exception:
                    pseudo_mask = None

            l_loc = localization_loss(pred_mask, mid, has_mask, pseudo_mask=pseudo_mask)

            # FIX-2: Correct consistency loss
            l_cons = torch.tensor(0.0, device=device)
            if Fs is not None and Ff is not None and Fn is not None:
                l_cons = branch_consistency_loss(Fs, Ff, Fn)

            # Temporal consistency
            mask_seq = out.get("mask_sequence")
            if mask_seq is None:
                mask_seq = pred_mask.unsqueeze(1).repeat(1, frames.shape[1], 1, 1, 1)
            l_temp = temporal_consistency_loss(mask_seq)

            loss = l_cls + loc_weight * l_loc + 0.05 * l_cons + 0.05 * l_temp

        # NaN/Inf detection — check BEFORE backward to avoid grad_fn crash.
        # The requires_grad error happens when a component is NaN/Inf and gets
        # replaced with a detached zero — the total loss then has no grad_fn.
        # Correct fix: skip the entire batch if loss is invalid.
        if not torch.isfinite(loss):
            nan_count += 1
            optimizer.zero_grad(set_to_none=True)
            # DO NOT call scaler.update() here.
            # GradScaler.update() asserts that scale()+backward() were called first.
            # Calling update() on a skipped batch → AssertionError: "No inf checks
            # were recorded prior to update." (the exact crash you hit at epoch 4).
            # The scaler state stays consistent on its own when we skip a batch —
            # it only needs update() after an actual backward pass.
            #
            # ADDITIONALLY: reset the scaler's loss scale when NaN appears.
            # A bad scale factor (e.g. 65536 after long stable training) can cause
            # gradient overflow on the next batch too, creating a cascade.
            # Halving the scale manually breaks the cascade.
            if hasattr(scaler, '_scale') and scaler._scale is not None:
                scaler._scale.fill_(scaler._scale.item() / 2.0)
            if nan_count > 10:
                raise RuntimeError(
                    f"Too many non-finite loss batches ({nan_count}). "
                    f"Last: cls={l_cls.item():.4f} loc={l_loc.item():.4f} "
                    f"cons={l_cons.item():.4f} temp={l_temp.item():.4f}"
                )
            continue

        # CORRECT AMP gradient clipping order
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Skip optimizer step if backward produced inf/nan gradients
        if torch.isfinite(grad_norm) and grad_norm < 50.0:
            scaler.step(optimizer)
        else:
            # Inf/NaN gradients after clipping = scale too high, halve it
            if hasattr(scaler, '_scale') and scaler._scale is not None:
                scaler._scale.fill_(max(scaler._scale.item() / 4.0, 64.0))
            optimizer.zero_grad(set_to_none=True)

        scaler.update()

        # Periodic VRAM defrag — prevents the incremental fragmentation
        # that causes OOM at epoch 2+ even when total usage looks fine
        if batch_idx % 500 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()

        # NEW-5: EMA update after each step
        if ema is not None:
            ema.update(model)

        bs = frames.size(0)
        with torch.no_grad():
            probs = torch.sigmoid(pred_cls.detach()).squeeze(1)
        overall, real_acc, fake_acc = batch_per_class_accuracy(pred_cls.detach(), labels if not do_mixup else labels_a)
        running_loss    += loss.item() * bs
        running_correct += overall * bs
        running_real_acc += real_acc * bs
        running_fake_acc += fake_acc * bs
        total_samples   += bs
        all_probs.extend(probs.cpu().tolist())
        # Use hard (rounded) labels for AUC computation
        hard_labels_for_log = labels.round().squeeze(1)
        all_labels_flat.extend(hard_labels_for_log.cpu().tolist())

        progress.set_postfix(
            loss=f"{running_loss/total_samples:.4f}",
            acc=f"{100*running_correct/total_samples:.1f}%",
            real=f"{100*running_real_acc/total_samples:.1f}%",
            fake=f"{100*running_fake_acc/total_samples:.1f}%",
            nan=nan_count,
        )

    progress.close()
    if total_samples == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "auc": 0.0, "nan_count": nan_count}

    # Compute training AUC
    try:
        train_auc = roc_auc_score(
            [round(l) for l in all_labels_flat],  # round mixed labels to nearest int
            all_probs
        )
    except Exception:
        train_auc = 0.0

    return {
        "loss":     running_loss / total_samples,
        "accuracy": running_correct / total_samples,
        "real_acc": running_real_acc / total_samples,
        "fake_acc": running_fake_acc / total_samples,
        "auc":      train_auc,
        "nan_count": nan_count,
    }


# ===========================================================================
# VALIDATION STEP
# ===========================================================================

def validate(model, loader, device, epoch, total_epochs, use_amp, cls_loss_fn):
    model.eval()

    running_loss    = 0.0
    running_loss_n  = 0     # separate counter for finite-loss batches only
    running_correct = 0.0
    total_samples   = 0.0
    all_probs       = []
    all_labels_flat = []
    all_preds       = []
    domain_correct  = defaultdict(list)
    domain_probs    = defaultdict(list)   # NEW: per-domain probs for AUC
    domain_labels   = defaultdict(list)   # NEW: per-domain labels for AUC

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [VAL]",
                    leave=True, dynamic_ncols=True)

    with torch.no_grad():
        for batch in progress:
            frames   = batch["frames"].to(device)
            labels   = batch["label"].to(device).unsqueeze(1)
            masks    = batch["mask_frames"].to(device)
            has_mask = batch["has_mask"].to(device)
            if has_mask.dtype != torch.bool:
                has_mask = has_mask.bool()
            domains  = batch.get("domain", [])

            with torch.cuda.amp.autocast(enabled=use_amp):
                out       = model(frames)
                pred_cls  = out["pred_cls"]
                pred_mask = out["pred_mask"]
                mid       = masks[:, masks.shape[1] // 2].unsqueeze(1)

                with torch.cuda.amp.autocast(enabled=False):
                    l_cls = cls_loss_fn(pred_cls.float(), labels.float())

                l_loc = localization_loss(pred_mask, mid, has_mask)
                loss  = l_cls + 0.5 * l_loc

            bs = frames.size(0)
            probs = torch.sigmoid(pred_cls.detach()).squeeze(1)
            # Replace any NaN probabilities with 0.5 (neutral prediction)
            probs = torch.where(torch.isfinite(probs), probs, torch.full_like(probs, 0.5))
            preds = (probs >= 0.5).float()
            correct = (preds == labels.round().squeeze(1)).float()

            # Only add loss to running total if it's finite
            if torch.isfinite(loss):
                running_loss   += loss.item() * bs
                running_loss_n += bs

            running_correct += correct.mean().item() * bs
            total_samples   += bs

            all_probs.extend(probs.cpu().tolist())
            all_labels_flat.extend(labels.squeeze(1).round().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            # Per-domain tracking (accuracy + AUC)
            for i, domain in enumerate(domains):
                domain_correct[domain].append(correct[i].item())
                domain_probs[domain].append(probs[i].item())
                domain_labels[domain].append(labels.squeeze(1)[i].item())

            progress.set_postfix(
                loss=f"{running_loss/running_loss_n:.4f}" if running_loss_n > 0 else "nan",
                acc=f"{100*running_correct/total_samples:.1f}%" if total_samples > 0 else "nan",
            )

    progress.close()
    if total_samples == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "auc": 0.0, "domain_stats": {}}

    try:
        auc = roc_auc_score(all_labels_flat, all_probs)
    except Exception:
        auc = 0.0

    # Per-domain stats
    # NOTE: Fake domains (Deepfakes, Face2Face etc.) only contain fake samples.
    # Real domains (youtube, actors) only contain real samples.
    # roc_auc_score requires both classes → shows n/a for single-class domains.
    # Instead, show: fake domains → Recall (% of fakes caught)
    #                real domains → Specificity (% of reals correctly identified)
    #                Mixed domains (DeepFakeDetection has both) → full AUC
    domain_stats = {}
    for d in domain_correct:
        n       = len(domain_correct[d])
        d_acc   = sum(domain_correct[d]) / n
        d_lbls  = domain_labels[d]
        d_prbs  = domain_probs[d]

        n_real  = sum(1 for l in d_lbls if l < 0.5)
        n_fake  = sum(1 for l in d_lbls if l >= 0.5)

        if n_real > 0 and n_fake > 0:
            # Mixed domain — full AUC is valid
            try:
                d_auc = roc_auc_score(d_lbls, d_prbs)
            except Exception:
                d_auc = float("nan")
            d_type = "mixed"
        elif n_fake > 0:
            # All-fake domain: show recall = fraction of fakes detected
            preds_domain = [1.0 if p > 0.5 else 0.0 for p in d_prbs]
            d_auc  = sum(preds_domain) / len(preds_domain)   # = recall on this domain
            d_type = "recall"
        else:
            # All-real domain: show specificity = fraction of reals correctly kept
            preds_domain = [1.0 if p <= 0.5 else 0.0 for p in d_prbs]
            d_auc  = sum(preds_domain) / len(preds_domain)
            d_type = "specificity"

        domain_stats[d] = {
            "acc": d_acc, "auc": d_auc, "n": n,
            "n_real": n_real, "n_fake": n_fake, "metric_type": d_type,
        }

    return {
        "loss":         running_loss / running_loss_n if running_loss_n > 0 else float("nan"),
        "accuracy":     running_correct / total_samples,
        "auc":          auc,
        "domain_stats": domain_stats,
        "labels":       all_labels_flat,
        "probs":        all_probs,
        "preds":        all_preds,
    }


# ===========================================================================
# MAIN TRAIN FUNCTION
# ===========================================================================

def train(
    model,
    train_dataset,
    val_dataset=None,
    epochs: int = 20,
    batch_size: int = 4,
    device=None,
    lr: float = 3e-4,
    backbone_lr_factor: float = 0.1,
    weight_decay: float = 1e-3,     # increased from 1e-4 → stronger L2 regularization
    num_workers: int = 0,
    patience: int = 7,
    warmup_epochs: int = 2,
    save_dir: str = "checkpoints",
    log_dir: str = "experiments/logs",
    use_amp: bool = True,
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    use_ema: bool = True,
    ema_decay: float = 0.9999,
    use_alt_freezing: bool = False,   # enable AltFreezing (better but changes training)
    resume_epoch: int = 0,            # skip epochs already completed (for resuming)
):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device  = torch.device(device)
    use_amp = use_amp and (device.type == "cuda")

    # ---- Enable training mode augmentation ----
    # This sets the dataset's augmentation flag so __getitem__ applies augmentation
    def _set_training_mode(ds, mode: bool):
        base = ds.dataset if hasattr(ds, "dataset") else ds
        if hasattr(base, "training_mode"):
            base.training_mode = mode

    _set_training_mode(train_dataset, True)
    if val_dataset is not None:
        _set_training_mode(val_dataset, False)

    # ---- Fast balanced sampler (FIX-1) ----
    sampler      = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,   # avoid single-sample batches (can cause BN issues)
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

    model = model.to(device)

    # ---- Loss ----
    cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # ---- Optimizer: separate LR for pretrained backbone ----
    backbone_params = list(model.backbone.parameters()) if hasattr(model, "backbone") else []
    backbone_ids    = {id(p) for p in backbone_params}
    other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]

    if backbone_params:
        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_factor, "name": "backbone"},
            {"params": other_params,    "lr": lr,                       "name": "rest"},
        ]
        print(f"\nLR groups: backbone={lr * backbone_lr_factor:.1e}, rest={lr:.1e}")
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    # CosineAnnealingLR: smooth decay over all epochs, NO restarts.
    # CosineAnnealingWarmRestarts caused NaN at restart points (epoch 7, 9)
    # because LR jumps back to max while GradScaler is still at high scale.
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,   # decay over remaining epochs after warmup
        eta_min=1e-6,
    )
    scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs,
                                after_scheduler=cosine_sched)

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp,
        init_scale=1024.0,       # Low start — grows slowly, stays safe through 30+ epochs
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,    # Only doubles every 2000 clean steps (not 500)
    )

    # ---- EMA (NEW-5) ----
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    # ---- AltFreezing (NEW-3) ----
    alt_freezing = None
    if use_alt_freezing:
        try:
            from models.afag_net_v3 import AltFreezingTrainer
            alt_freezing = AltFreezingTrainer(model)
            print("AltFreezing enabled.")
        except ImportError:
            print("AltFreezing: AFAGNetV3 not found, skipping.")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    best_path    = os.path.join(save_dir, "best_model.pth")
    latest_path  = os.path.join(save_dir, "latest_model.pth")
    ema_path     = os.path.join(save_dir, "ema_model.pth")
    log_path     = os.path.join(log_dir,  "epoch_results_v3.txt")

    # NEW-6: Save on val AUC instead of val loss
    best_auc         = 0.0
    patience_counter = 0
    history          = defaultdict(list)

    # Fast-forward scheduler to resume_epoch position
    if resume_epoch > 0:
        print(f"Resuming training from epoch {resume_epoch + 1}")
        for _ in range(resume_epoch):
            scheduler.step(0)   # advance internal counter; warmup uses epoch arg separately
        best_auc = 0.0  # will be updated by first val run

    print_system_info(device, model)
    print(f"\n{'='*70}")
    print(f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset) if val_loader else 'N/A'}")
    print(f"AMP: {use_amp} | MixUp: {use_mixup} | EMA: {use_ema} | AltFreezing: {use_alt_freezing}")
    if resume_epoch > 0:
        print(f"Resuming from epoch {resume_epoch} — training epochs {resume_epoch+1}–{epochs}")
    print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(resume_epoch, epochs):
        epoch_start = time.time()
        scheduler.step(epoch)   # warmup scheduler still needs epoch for LR scaling

        # AltFreezing phase switch
        if alt_freezing is not None:
            alt_freezing.set_epoch(epoch)

        # ---- Train ----
        train_m = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, scaler=scaler,
            epoch=epoch, total_epochs=epochs, use_amp=use_amp,
            cls_loss_fn=cls_loss_fn, warmup_epochs=warmup_epochs,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha, ema=ema,
        )

        # ---- Validate ----
        val_m = {}
        if val_loader is not None:
            torch.cuda.empty_cache()
            eval_model = ema.model if ema is not None else model
            val_m = validate(
                model=eval_model, loader=val_loader, device=device,
                epoch=epoch, total_epochs=epochs,
                use_amp=use_amp, cls_loss_fn=cls_loss_fn,
            )

        elapsed  = time.time() - epoch_start
        val_auc  = val_m.get("auc", 0.0)
        val_loss = val_m.get("loss", float("nan"))
        val_acc  = val_m.get("accuracy", 0.0)

        # Log — full history for graph generation
        history["train_loss"].append(float(train_m["loss"]) if not math.isnan(train_m["loss"]) else float("nan"))
        history["train_auc"].append(float(train_m.get("auc", 0.0)))
        history["train_acc"].append(float(train_m.get("accuracy", 0.0)))
        history["train_real_acc"].append(float(train_m.get("real_acc", 0.0)))
        history["train_fake_acc"].append(float(train_m.get("fake_acc", 0.0)))
        history["train_nan"].append(int(train_m.get("nan_count", 0)))
        history["val_loss"].append(float(val_loss) if not math.isnan(val_loss) else float("nan"))
        history["val_auc"].append(float(val_auc))
        history["val_acc"].append(float(val_acc))
        history["lr_backbone"].append(float(optimizer.param_groups[0]["lr"]))
        history["lr_rest"].append(float(optimizer.param_groups[-1]["lr"]))
        history["epoch"].append(epoch + 1)

        print(f"\n{'-'*70}")
        print(f"Epoch {epoch+1}/{epochs} Summary")
        print(f"{'-'*70}")
        print(f"Train  Loss: {train_m['loss']:.5f}  Acc: {train_m['accuracy']*100:.1f}%  "
              f"AUC: {train_m.get('auc',0):.4f}  "
              f"[Real: {train_m.get('real_acc',0)*100:.1f}%  Fake: {train_m.get('fake_acc',0)*100:.1f}%]  "
              f"NaN: {train_m.get('nan_count',0)}")
        if val_m:
            print(f"Val    Loss: {val_loss:.5f}  Acc: {val_acc*100:.1f}%  AUC: {val_auc:.4f}")
            domain_stats = val_m.get("domain_stats", {})
            if domain_stats:
                print(f"  {'Domain':<22} {'N':>5}  {'Acc':>7}  {'Metric':>12}  {'Type'}")
                print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*12}  {'-'*12}")
                for domain, stats in sorted(domain_stats.items()):
                    mtype = stats.get("metric_type", "?")
                    mval  = stats.get("auc", float("nan"))
                    if mtype == "mixed":
                        metric_label = f"AUC={mval:.4f}"
                    elif mtype == "recall":
                        metric_label = f"Recall={mval:.4f}"
                    elif mtype == "specificity":
                        metric_label = f"Specif={mval:.4f}"
                    else:
                        metric_label = f"{mval:.4f}"
                    print(f"  {domain:<22} {stats['n']:>5}  {stats['acc']*100:>6.1f}%  {metric_label:>12}  ({stats['n_real']}R/{stats['n_fake']}F)")
        lr_str = " | ".join(f"{g.get('name','?')}:{g['lr']:.1e}" for g in optimizer.param_groups)
        print(f"LR: {lr_str}   Time: {elapsed:.1f}s")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch={epoch+1},train_loss={train_m['loss']:.6f},"
                    f"train_auc={train_m.get('auc',0):.4f},"
                    f"val_loss={val_loss:.6f},val_auc={val_auc:.4f},"
                    f"val_acc={val_acc:.4f},nan={train_m.get('nan_count',0)},"
                    f"time={elapsed:.1f}\n")

        # NEW-6: Save on val AUC
        # NEW-7: Don't count warmup epochs in patience
        epoch_had_nan = train_m.get("nan_count", 0) > 0
        if epoch < warmup_epochs:
            print(f"  -> Warmup epoch, not counting toward patience.")
            torch.save(model.state_dict(), latest_path)
            if ema is not None and not epoch_had_nan:
                torch.save(ema.model.state_dict(), ema_path)
            continue

        # Only save best/EMA if epoch was clean (no NaN batches) AND AUC improved
        # This prevents a corrupted EMA from overwriting a good checkpoint
        if not math.isnan(val_auc) and val_auc > best_auc and not epoch_had_nan:
            best_auc         = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            if ema is not None:
                torch.save(ema.model.state_dict(), ema_path)
            print(f"  -> BEST model saved (val AUC: {val_auc:.4f})\n")
        elif epoch_had_nan:
            patience_counter += 1
            print(f"  -> Skipped save (epoch had {train_m.get('nan_count',0)} NaN batch(es)). "
                  f"Best AUC stays: {best_auc:.4f}. Patience: {patience_counter}/{patience}\n")
        else:
            patience_counter += 1
            print(f"  -> No improvement (best AUC: {best_auc:.4f}). Patience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"{'='*70}")
                print(f"EARLY STOPPING after {patience} epochs without AUC improvement")
                print(f"{'='*70}\n")
                break

        torch.save(model.state_dict(), latest_path)
        # Only update latest EMA if epoch was clean
        if ema is not None and not epoch_had_nan:
            torch.save(ema.model.state_dict(), ema_path)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Time  : {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Saved: {best_path}")
    if ema is not None:
        print(f"EMA  : {ema_path}")
    print(f"Logs : {log_path}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Save training graphs automatically after training
    graph_dir = os.path.join(log_dir, "graphs")
    try:
        plot_training_history(dict(history), save_dir=graph_dir, batch_size=batch_size)
    except Exception as e:
        print(f"[WARN] Graph generation failed: {e}")

    return dict(history)