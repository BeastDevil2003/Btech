"""
train_pipeline_final.py  — AFAGNet Improved Training Pipeline
=============================================================
Critical fixes from v1:
  FIX-1: scaler.unscale_() BEFORE clip_grad_norm_ (broken gradient clipping was the PRIMARY NaN cause)
  FIX-2: pos_weight replaced with Focal Loss (pos_weight=0.217 caused "always predict real" → 8.34% val)
  FIX-3: NaN batch detection and skip (prevents weight corruption after a NaN loss)
  FIX-4: Lower clip threshold 5.0 → 1.0 (5.0 was too permissive even when clipping worked)
  FIX-5: Cosine LR schedule + linear warmup (ReduceLROnPlateau was too reactive)
  FIX-6: Separate LR for pretrained backbone (prevents destroying pretrained features)
  FIX-7: Per-class accuracy logging (was hiding the "all real" collapse)
"""

import os
import time
import math
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

from train.psuedo_mask import generate_pseudo_mask

warnings.filterwarnings("ignore", message=".*flash attention.*",             category=UserWarning)
warnings.filterwarnings("ignore", message=".*ComplexHalf support.*",         category=UserWarning)
warnings.filterwarnings("ignore", message=".*rcond parameter will change.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*HF Hub.*",                      category=UserWarning)

l1 = nn.L1Loss()


# =============================================================================
# FOCAL LOSS  (replaces BCEWithLogitsLoss + pos_weight)
#
# Why Focal Loss instead of weighted BCE?
#   - pos_weight=0.217 in v1 downweighted fake → "always predict real" collapse
#   - Focal Loss auto-focuses on hard examples regardless of class ratio
#   - gamma=2.0 is the standard; alpha=0.5 for balanced batches
#   - Numerically equivalent to BCE when gamma=0
#
# With WeightedRandomSampler (50/50 batches):
#   alpha=0.5 treats both classes equally in expectation
#   gamma=2.0 down-weights easy predictions (confident correct → small loss)
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # bce_loss per element (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t: probability of the true class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        # alpha_t: per-class weight
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        # Focal weight
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# BALANCED SAMPLER  (unchanged — still needed to equalize batch composition)
# =============================================================================
def make_balanced_sampler(train_dataset):
    labels = [int(train_dataset[i]["label"].item()) for i in range(len(train_dataset))]
    counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / (counts[lbl] + 1e-6) for lbl in labels]
    n_real, n_fake = counts[0], counts[1]
    print(f"\nClass balance  — real: {n_real}, fake: {n_fake}")
    print(f"Balanced sampler — real w={1/n_real:.5f}, fake w={1/n_fake:.5f}")
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ---------------------------
# OUTPUT HELPERS
# ---------------------------
def unpack_model_output(out):
    if isinstance(out, dict):
        return out
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            pm, feats, attn = out
            return {"pred_mask": pm, "features": feats, "attention": attn}
        if len(out) == 2:
            pc, pm = out
            return {"pred_cls": pc, "pred_mask": pm}
    raise TypeError(f"Unsupported model output type: {type(out)}")


def get_pseudo_features(out):
    fs, ff, fn = out.get("Fs"), out.get("Ff"), out.get("Fn")
    if fs is not None and ff is not None and fn is not None:
        return fs, ff, fn
    fb = out.get("Ffusion") or out.get("features") or out.get("pred_mask")
    if fb is not None and fb.dim() == 4:
        return fb, fb, fb
    return None, None, None


# ---------------------------
# LOSS FUNCTIONS
# ---------------------------
def temporal_loss(mask_seq):
    return torch.abs(mask_seq[:, 1:] - mask_seq[:, :-1]).mean()


def localization_loss(pred_mask, gt_mask, has_mask, pseudo_mask=None):
    real_loss   = torch.tensor(0.0, device=pred_mask.device)
    pseudo_loss = torch.tensor(0.0, device=pred_mask.device)
    if has_mask.sum() > 0:
        real_loss = l1(pred_mask[has_mask], gt_mask[has_mask])
    if (~has_mask).sum() > 0 and pseudo_mask is not None:
        pseudo_loss = l1(pred_mask[~has_mask], pseudo_mask[~has_mask])
    return real_loss + 0.3 * pseudo_loss


def batch_accuracy(pred_logits, labels):
    probs = torch.sigmoid(pred_logits)
    preds = (probs >= 0.5).float()
    return (preds == labels).float().mean().item()


def batch_per_class_accuracy(pred_logits, labels):
    """Returns (overall_acc, real_acc, fake_acc) — catches the 'all real' collapse."""
    probs = torch.sigmoid(pred_logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).float()
    real_mask = (labels == 0).float()
    fake_mask = (labels == 1).float()
    overall = correct.mean().item()
    real_acc = (correct * real_mask).sum() / (real_mask.sum() + 1e-6)
    fake_acc = (correct * fake_mask).sum() / (fake_mask.sum() + 1e-6)
    return overall, real_acc.item(), fake_acc.item()


# ---------------------------
# WARMUP SCHEDULER
# ---------------------------
class WarmupScheduler:
    """
    Linear warmup for the first `warmup_epochs` epochs.
    After warmup, hands control to `after_scheduler` (CosineAnnealingWarmRestarts).
    """
    def __init__(self, optimizer, warmup_epochs: int, after_scheduler):
        self.optimizer        = optimizer
        self.warmup_epochs    = warmup_epochs
        self.after_scheduler  = after_scheduler
        self.base_lrs         = [g["lr"] for g in optimizer.param_groups]
        self._epoch           = 0

    def step(self, epoch: int):
        self._epoch = epoch
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base_lr * scale
        else:
            self.after_scheduler.step(epoch - self.warmup_epochs)

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


# ---------------------------
# LOGGING
# ---------------------------
def print_system_info(device, model):
    print("\n" + "=" * 70)
    print("SYSTEM & MODEL INFORMATION")
    print("=" * 70)
    if device.type == "cuda":
        print(f"Device:          GPU (CUDA)")
        print(f"GPU Model:       {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory:      {mem:.2f} GB")
        print(f"PyTorch CUDA:    {torch.version.cuda}")
    else:
        print("Device:          CPU")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PyTorch:         {torch.__version__}")
    print(f"AFAGNet params:  {total:,} total / {trainable:,} trainable")
    print("=" * 70 + "\n")


def print_training_config(cfg):
    print("=" * 70)
    print("TRAINING CONFIGURATION v2")
    print("=" * 70)
    for k, v in cfg.items():
        print(f"  {k:<34}: {v}")
    print("=" * 70 + "\n")


# ---------------------------
# TRAINING STEP
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, scaler,
                    epoch, total_epochs, use_amp, cls_loss_fn,
                    warmup_epochs=2):
    model.train()
    running_loss    = 0.0
    running_correct = 0.0
    running_real_acc  = 0.0
    running_fake_acc  = 0.0
    total_samples   = 0.0
    nan_count       = 0
    high_grad_count = 0

    loc_weight = max(0.3, 1.0 - epoch * 0.15)

    progress = tqdm(loader,
                    desc=f"Epoch {epoch+1}/{total_epochs} [TRAIN]",
                    leave=True, dynamic_ncols=True)

    for batch_idx, batch in enumerate(progress):
        frames   = batch["frames"].to(device)
        labels   = batch["label"].to(device).unsqueeze(1)
        masks    = batch["mask_frames"].to(device)
        has_mask = batch["has_mask"].to(device).bool()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out       = unpack_model_output(model(frames))
            pred_cls  = out["pred_cls"]
            pred_mask = out["pred_mask"]

            fs, ff, fn = get_pseudo_features(out)
            pseudo_mid = None
            if fs is not None:
                ps = generate_pseudo_mask(model, frames, fs, ff, fn, use_gradcam=False)
                pseudo_mid = ps[:, ps.shape[1] // 2]

            mid = masks[:, masks.shape[1] // 2].unsqueeze(1)

            with torch.cuda.amp.autocast(enabled=False):
                l_cls = cls_loss_fn(pred_cls.float(), labels.float())

            l_loc = localization_loss(pred_mask, mid, has_mask, pseudo_mid)

            l_cons = torch.tensor(0.0, device=device)
            if out.get("Csf") is not None:
                l_cons = l_cons + (1.0 - torch.sigmoid(out["Csf"])).mean()
            if out.get("Csn") is not None:
                l_cons = l_cons + (1.0 - torch.sigmoid(out["Csn"])).mean()

            mask_seq = out.get("mask_sequence")
            if mask_seq is None:
                mask_seq = pred_mask.unsqueeze(1).repeat(1, frames.shape[1], 1, 1, 1)
            l_temp = temporal_loss(mask_seq)

            loss = l_cls + loc_weight * l_loc + 0.1 * l_temp + 0.1 * l_cons

        # =====================================================================
        # FIX-1: Correct AMP gradient clipping order
        #
        # OLD (WRONG):
        #   scaler.scale(loss).backward()
        #   clip_grad_norm_(model.parameters(), 5.0)  ← clips SCALED grads!
        #   scaler.step(); scaler.update()
        #
        # NEW (CORRECT):
        #   scaler.scale(loss).backward()
        #   scaler.unscale_(optimizer)               ← unscale FIRST
        #   clip_grad_norm_(model.parameters(), 1.0) ← now clips real gradients
        #   scaler.step(); scaler.update()
        #
        # Without unscale_(), clip at 5.0 on gradients scaled by 65536
        # is effectively clipping at 0.000076 — essentially zero.
        # Gradients passed to optimizer were ALWAYS unclipped in v1.
        # =====================================================================

        # FIX-3: NaN detection — skip batch before it corrupts weights
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(nan=f"{nan_count}", batch=f"{batch_idx+1}/{len(loader)}")
            if nan_count > 20:
                raise RuntimeError(
                    f"Too many NaN batches ({nan_count}). "
                    "Check frequency branch float16 usage and CGAF temperature."
                )
            continue

        scaler.scale(loss).backward()

        # FIX-1 (continued): unscale before clipping
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Monitor for gradient explosions (useful diagnostic)
        if total_norm > 5.0:
            high_grad_count += 1

        scaler.step(optimizer)
        scaler.update()

        bs = frames.size(0)
        overall_acc, real_acc, fake_acc = batch_per_class_accuracy(pred_cls.detach(), labels)
        running_loss    += loss.item() * bs
        running_correct += overall_acc * bs
        running_real_acc  += real_acc * bs
        running_fake_acc  += fake_acc * bs
        total_samples   += bs

        progress.set_postfix(
            loss=f"{running_loss/total_samples:.4f}",
            acc=f"{100*running_correct/total_samples:.1f}%",
            real_acc=f"{100*running_real_acc/total_samples:.1f}%",
            fake_acc=f"{100*running_fake_acc/total_samples:.1f}%",
            lw=f"{loc_weight:.2f}",
            nan=nan_count,
            batch=f"{batch_idx+1}/{len(loader)}",
        )

    progress.close()
    n = len(loader.dataset)
    if total_samples == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "real_acc": 0.0, "fake_acc": 0.0}

    result = {
        "loss":     running_loss    / total_samples,
        "accuracy": running_correct / total_samples,
        "real_acc": running_real_acc  / total_samples,
        "fake_acc": running_fake_acc  / total_samples,
        "nan_count": nan_count,
        "high_grad_count": high_grad_count,
    }
    return result


# ---------------------------
# VALIDATION STEP
# ---------------------------
def validate(model, loader, device, epoch, total_epochs, use_amp, cls_loss_fn):
    model.eval()
    running_loss    = 0.0
    running_correct = 0.0
    running_real_acc  = 0.0
    running_fake_acc  = 0.0
    total_samples   = 0.0

    progress = tqdm(loader,
                    desc=f"Epoch {epoch+1}/{total_epochs} [VAL]",
                    leave=True, dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            frames   = batch["frames"].to(device)
            labels   = batch["label"].to(device).unsqueeze(1)
            masks    = batch["mask_frames"].to(device)
            has_mask = batch["has_mask"].to(device).bool()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out       = unpack_model_output(model(frames))
                pred_cls  = out["pred_cls"]
                pred_mask = out["pred_mask"]

                mid = masks[:, masks.shape[1] // 2].unsqueeze(1)

                with torch.cuda.amp.autocast(enabled=False):
                    l_cls = cls_loss_fn(pred_cls.float(), labels.float())

                l_loc = localization_loss(pred_mask, mid, has_mask, pseudo_mask=None)

                l_cons = torch.tensor(0.0, device=device)
                if out.get("Csf") is not None:
                    l_cons = l_cons + (1.0 - torch.sigmoid(out["Csf"])).mean()
                if out.get("Csn") is not None:
                    l_cons = l_cons + (1.0 - torch.sigmoid(out["Csn"])).mean()

                mask_seq = out.get("mask_sequence")
                if mask_seq is None:
                    mask_seq = pred_mask.unsqueeze(1).repeat(1, frames.shape[1], 1, 1, 1)
                l_temp = temporal_loss(mask_seq)

                loss = l_cls + 0.5 * l_loc + 0.1 * l_temp + 0.1 * l_cons

            if torch.isnan(loss):
                continue

            bs = frames.size(0)
            overall_acc, real_acc, fake_acc = batch_per_class_accuracy(pred_cls.detach(), labels)
            running_loss    += loss.item() * bs
            running_correct += overall_acc * bs
            running_real_acc  += real_acc * bs
            running_fake_acc  += fake_acc * bs
            total_samples   += bs

            progress.set_postfix(
                loss=f"{running_loss/total_samples:.4f}" if total_samples > 0 else "nan",
                acc=f"{100*running_correct/total_samples:.1f}%" if total_samples > 0 else "nan",
                real_acc=f"{100*running_real_acc/total_samples:.1f}%",
                fake_acc=f"{100*running_fake_acc/total_samples:.1f}%",
                batch=f"{batch_idx+1}/{len(loader)}",
            )

    progress.close()
    if total_samples == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "real_acc": 0.0, "fake_acc": 0.0}

    return {
        "loss":     running_loss    / total_samples,
        "accuracy": running_correct / total_samples,
        "real_acc": running_real_acc  / total_samples,
        "fake_acc": running_fake_acc  / total_samples,
    }


# ---------------------------
# MAIN TRAIN FUNCTION
# ---------------------------
def train(
    model,
    train_dataset,
    val_dataset=None,
    epochs=20,
    batch_size=2,
    device=None,
    lr=3e-4,
    backbone_lr_factor=0.1,     # backbone LR = lr * backbone_lr_factor
    weight_decay=1e-4,
    num_workers=0,
    patience=7,
    warmup_epochs=2,
    save_dir="checkpoints",
    log_dir="experiments/logs",
    use_amp=True,
    focal_alpha=0.5,
    focal_gamma=2.0,
):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device  = torch.device(device)
    use_amp = use_amp and (device.type == "cuda")

    cfg = {
        "batch_size":            batch_size,
        "epochs":                epochs,
        "lr":                    lr,
        "backbone_lr":           lr * backbone_lr_factor,
        "weight_decay":          weight_decay,
        "patience":              patience,
        "warmup_epochs":         warmup_epochs,
        "use_amp":               use_amp,
        "cls_loss":              f"FocalLoss(alpha={focal_alpha}, gamma={focal_gamma})",
        "class_balance":         "WeightedRandomSampler (50/50 batches)",
        "loc_weight":            "1.0 decaying 0.15/epoch to floor 0.3",
        "grad_clip":             "1.0 (applied after scaler.unscale_)",
        "scheduler":             f"CosineAnnealingWarmRestarts(T_0=5, T_mult=2) + {warmup_epochs}ep warmup",
        "consistency_loss":      "(1 - sigmoid(Csf)) rewards branch alignment",
        "NaN_resilience":        "skip batch, log count, raise after 20",
        "save_dir":              save_dir,
        "log_dir":               log_dir,
    }

    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

    model = model.to(device)

    # FIX-2: Focal Loss instead of BCE + wrong pos_weight
    cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # FIX-6: Separate LR groups — pretrained backbone gets lower LR
    backbone_params = list(model.backbone.parameters()) if hasattr(model, "backbone") else []
    backbone_ids    = set(id(p) for p in backbone_params)
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

    # FIX-5: CosineAnnealingWarmRestarts + warmup
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs,
                                after_scheduler=cosine_scheduler)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [], "train_real_acc": [], "train_fake_acc": [],
        "val_loss":   [], "val_acc":   [], "val_real_acc":   [], "val_fake_acc":   [],
        "best_epoch": 0,
    }

    best_metric      = float("inf")
    best_path        = os.path.join(save_dir, "best_model.pth")
    latest_path      = os.path.join(save_dir, "latest_model.pth")
    history_path     = os.path.join(log_dir,  "train_history.pt")
    epoch_log_path   = os.path.join(log_dir,  "epoch_results_v2.txt")
    patience_counter = 0

    print("\n" + "=" * 70)
    print(f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples:   {len(val_loader.dataset)}")
    print()
    print_system_info(device, model)
    print_training_config(cfg)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        scheduler.step(epoch)

        train_m = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, scaler=scaler, epoch=epoch,
            total_epochs=epochs, use_amp=use_amp, cls_loss_fn=cls_loss_fn,
            warmup_epochs=warmup_epochs,
        )
        train_loss = train_m["loss"]
        train_acc  = train_m["accuracy"]
        metric_loss = train_loss
        val_loss = val_acc = None

        if val_loader is not None:
            torch.cuda.empty_cache()
            val_m = validate(
                model=model, loader=val_loader, device=device,
                epoch=epoch, total_epochs=epochs,
                use_amp=use_amp, cls_loss_fn=cls_loss_fn,
            )
            val_loss, val_acc = val_m["loss"], val_m["accuracy"]
            metric_loss = val_loss if not math.isnan(val_loss) else train_loss

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_real_acc"].append(float(train_m.get("real_acc", 0)))
        history["train_fake_acc"].append(float(train_m.get("fake_acc", 0)))
        history["val_loss"].append(None if val_loss is None else float(val_loss))
        history["val_acc"].append(None if val_acc  is None else float(val_acc))
        if val_m:
            history["val_real_acc"].append(float(val_m.get("real_acc", 0)))
            history["val_fake_acc"].append(float(val_m.get("fake_acc", 0)))

        elapsed    = time.time() - epoch_start
        current_lrs = [g["lr"] for g in optimizer.param_groups]
        loc_w      = max(0.3, 1.0 - epoch * 0.15)

        print("\n" + "-" * 70)
        print(f"Epoch {epoch+1}/{epochs} Summary")
        print("-" * 70)
        print(f"Train Loss : {train_loss:.6f}   Train Acc: {train_acc*100:.2f}%  "
              f"[Real: {train_m.get('real_acc', 0)*100:.1f}%  Fake: {train_m.get('fake_acc', 0)*100:.1f}%]")
        if val_loss is not None:
            print(f"Val Loss   : {val_loss:.6f}   Val Acc:   {val_acc*100:.2f}%  "
                  f"[Real: {val_m.get('real_acc', 0)*100:.1f}%  Fake: {val_m.get('fake_acc', 0)*100:.1f}%]")
        lr_str = " | ".join([f"{g.get('name','?')}:{g['lr']:.1e}" for g in optimizer.param_groups])
        print(f"LR: {lr_str}   Loc weight: {loc_w:.2f}   NaN batches: {train_m.get('nan_count',0)}   Time: {elapsed:.1f}s")

        with open(epoch_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"epoch={epoch+1},train_loss={train_loss:.6f},"
                f"train_acc={train_acc:.4f},"
                f"train_real_acc={train_m.get('real_acc',0):.4f},"
                f"train_fake_acc={train_m.get('fake_acc',0):.4f},"
                f"val_loss={'None' if val_loss is None else f'{val_loss:.6f}'},"
                f"val_acc={'None' if val_acc is None else f'{val_acc:.4f}'},"
                f"nan_batches={train_m.get('nan_count',0)},"
                f"time={elapsed:.1f}\n"
            )

        if not math.isnan(metric_loss) and metric_loss < best_metric:
            best_metric           = metric_loss
            history["best_epoch"] = epoch + 1
            patience_counter      = 0
            torch.save(model.state_dict(), best_path)
            tag = "val" if val_loss is not None else "train"
            print(f"  -> BEST model saved ({tag} loss: {metric_loss:.6f})\n")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print("=" * 70)
                print(f"EARLY STOPPING after {patience} epochs without improvement")
                print("=" * 70 + "\n")
                break

    total_time = time.time() - start_time
    torch.save(model.state_dict(), latest_path)
    torch.save(history, history_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Time  : {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    print(f"Best Epoch  : {history['best_epoch']}   Best Loss: {best_metric:.6f}")
    print(f"Train Loss  : {history['train_loss'][-1]:.6f}   "
          f"Train Acc: {history['train_acc'][-1]*100:.2f}%")
    if history["val_loss"] and history["val_loss"][-1] is not None:
        print(f"Val Loss    : {history['val_loss'][-1]:.6f}   "
              f"Val Acc: {history['val_acc'][-1]*100:.2f}%")
    print(f"\nSaved: {best_path} | {latest_path}")
    print(f"Logs:  {epoch_log_path}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    return history
