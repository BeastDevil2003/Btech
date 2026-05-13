"""
run_train_v3.py  — Entry point for AFAGNet v3 training
=======================================================
Wires together:
  - FFPPDatasetV2 (face alignment + augmentation)
  - AFAGNetV3     (fixed imports, CGAFv2, dual-path spatial)
  - train_pipeline_v3 (MixUp, EMA, AUC-based saving, correct losses)

IMPORTANT: First run will build the v2 cache (face alignment).
This takes ~3-4 hours but only happens once.
"""

import os
import sys
import torch
import warnings

from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from data.loader.splits import build_identity_disjoint_split
from models.afag_net_v3 import AFAGNetV3
from torch.utils.data import Subset
from train.train_pipeline_v3 import train

warnings.filterwarnings("ignore", message=".*HF Hub.*",      category=UserWarning)
warnings.filterwarnings("ignore", message=".*rcond.*",        category=FutureWarning)


def main():
    root      = "FaceForensics_Data"
    cache_dir = "dataset/cache/ffpp_processed_v2"

    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)

    # ----------------------------------------------------------------
    # STEP 1: Build v2 cache using FFPPDatasetV2 directly
    # (The old build_cache.py uses FFPPDataset v1 — wrong class, wrong
    #  cache signature. Must use FFPPDatasetV2 to get the right .pt files.)
    # ----------------------------------------------------------------
    cache_builder = FFPPDatasetV2(
        video_paths=video_paths,
        labels=labels,
        mask_paths=mask_paths,
        domains=domains,
        cache_dir=cache_dir,
        use_alignment=True,
        training_mode=False,    # never augment during cache build
        label_smoothing=0.0,
        use_sbi=False,
        temporal_strategy="blend",
    )

    stats = cache_builder.get_cache_stats()
    print(f"Cache directory : {cache_dir}")
    print(f"Cache status    : {stats['cached']}/{stats['total']} cached, {stats['missing']} missing")

    if stats["missing"] > 0:
        print(f"\nBuilding {stats['missing']} missing cache entries (face detection + alignment).")
        print("This only runs once. Grab a coffee — ~3-4 hours for 9431 videos.")
        print("You can interrupt and restart; completed entries are not rebuilt.\n")
        cache_builder.init_face_detector()
        cache_builder.build_cache(num_workers=0)   # 0 workers = safe on Windows
        stats = cache_builder.get_cache_stats()
        print(f"\nCache complete: {stats['cached']}/{stats['total']}")

    # Guard: refuse to train if fewer than 80% of samples are cached
    # (training on live-detected frames is 50-100× slower per batch)
    cache_pct = stats["cached"] / max(stats["total"], 1)
    if cache_pct < 0.80:
        print(f"\nERROR: Only {stats['cached']}/{stats['total']} ({cache_pct*100:.1f}%) samples cached.")
        print("Training on uncached samples is extremely slow (5+ s/it vs 0.3 s/it).")
        print("Run this script again — cache building will resume where it left off.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # STEP 2: Create train / val datasets with correct flags
    # ----------------------------------------------------------------
    train_indices, val_indices = build_identity_disjoint_split(
        video_paths, val_ratio=0.2, seed=42
    )

    # Training dataset: augmentation ON, label smoothing ON
    train_dataset = Subset(
        FFPPDatasetV2(
            video_paths=video_paths, labels=labels,
            mask_paths=mask_paths, domains=domains,
            cache_dir=cache_dir,
            use_alignment=True,
            training_mode=True,      # ← augmentation ON
            label_smoothing=0.05,
            use_sbi=True,
            temporal_strategy="blend",
        ),
        train_indices,
    )

    # Validation dataset: augmentation OFF, no smoothing
    val_dataset = Subset(
        FFPPDatasetV2(
            video_paths=video_paths, labels=labels,
            mask_paths=mask_paths, domains=domains,
            cache_dir=cache_dir,
            use_alignment=True,
            training_mode=False,     # ← augmentation OFF
            label_smoothing=0.0,
            use_sbi=False,
            temporal_strategy="blend",
        ),
        val_indices,
    )

    print(f"\nIdentity-aware split: train={len(train_dataset)}, val={len(val_dataset)}")

    # ----------------------------------------------------------------
    # STEP 3: Model — load best checkpoint if it exists (resume)
    # ----------------------------------------------------------------
    # Set your target total epochs here. If resuming, training continues
    # from where it left off up to this number.
    TOTAL_EPOCHS = 30   # ← change this to train longer

    model = AFAGNetV3(use_gradient_checkpointing=False)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AFAGNetV3 params: {total:,} total / {trainable:,} trainable")

    checkpoint_path = "checkpoints/best_model.pth"
    resume_epoch    = 0
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        log_path = "experiments/logs/epoch_results_v3.txt"
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if lines:
                try:
                    resume_epoch = int(lines[-1].split("epoch=")[1].split(",")[0])
                except Exception:
                    resume_epoch = 0

        if resume_epoch >= TOTAL_EPOCHS:
            print(f"\nAlready completed {resume_epoch} epochs (target={TOTAL_EPOCHS}).")
            print(f"To train more, increase TOTAL_EPOCHS above {resume_epoch} in run_train_v3.py")
            return
        print(f"\nResuming from epoch {resume_epoch} → training epochs {resume_epoch+1}–{TOTAL_EPOCHS}")
    else:
        print("\nNo checkpoint found — training from scratch.")

    # ----------------------------------------------------------------
    # STEP 4: Train
    # ----------------------------------------------------------------
    history = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=TOTAL_EPOCHS,
        batch_size=2,
        num_workers=0,
        lr=3e-4,
        backbone_lr_factor=0.1,
        weight_decay=5e-4,         # balanced: 1e-4 caused overfit, 1e-3 caused underfit
        patience=7,
        warmup_epochs=2,
        use_amp=True,
        focal_alpha=0.5,
        focal_gamma=2.0,
        use_mixup=True,
        mixup_alpha=0.2,           # back to 0.2 — 0.4 was too aggressive for batch=2
        use_ema=True,
        ema_decay=0.9999,
        use_alt_freezing=False,
        save_dir="checkpoints",
        log_dir="experiments/logs",
        resume_epoch=resume_epoch,
    )

    print(f"\nBest Val AUC: {max(history.get('val_auc', [0])):.4f}")


if __name__ == "__main__":
    main()
