"""
─────────────────────────────────────────────────────────────
                    batch=2         batch=4         batch=8
─────────────────────────────────────────────────────────────
weight_decay        5e-4            5e-4            3e-4
mixup_alpha         0.2             0.3             0.4
lr                  3e-4            3e-4            4e-4
warmup_epochs       2               2               3
patience            7               7               5
num_workers         0               2               4
focal_alpha         0.25            0.25            0.25  (FIXED from 0.5)
─────────────────────────────────────────────────────────────
"""

import argparse
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

FORCE_BATCH_SIZE = None   # set to 2, 4, or 8 to override auto-detection when no CLI batch-size is provided
DEFAULT_ROOT = "FaceForensics_Data"
DEFAULT_CACHE_DIR = "dataset/cache/ffpp_processed_v2"

HPARAM_TABLE = {
    2: dict(weight_decay=5e-4, mixup_alpha=0.2, lr=3e-4,
            warmup_epochs=2, patience=7, num_workers=0, focal_alpha=0.25,
            note="Batch_Size_2"),
    4: dict(weight_decay=5e-4, mixup_alpha=0.3, lr=3e-4,
            warmup_epochs=2, patience=7, num_workers=2, focal_alpha=0.25,
            note="Batch_Size_4"),
    8: dict(weight_decay=3e-4, mixup_alpha=0.4, lr=4e-4,
            warmup_epochs=3, patience=5, num_workers=4, focal_alpha=0.25,
            note="Batch_Size_8"),
}


def detect_batch_size():                              ############################### for setup batch size
    """Detect available GPU memory and select the best batch size."""
    if not torch.cuda.is_available():
        return 2
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 14:
        return 8
    elif vram_gb >= 8:
        return 4
    return 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AFAGNet v3 with optional batch-size and HPARAM overrides.")
    parser.add_argument("--batch-size", type=int, choices=[2, 4, 8],
                        help="Override auto-selected batch size profile.")
    parser.add_argument("--weight-decay", type=float, help="Override weight_decay from selected profile.")
    parser.add_argument("--mixup-alpha", type=float, help="Override mixup_alpha from selected profile.")
    parser.add_argument("--lr", type=float, help="Override learning rate from selected profile.")
    parser.add_argument("--warmup-epochs", type=int, help="Override warmup_epochs from selected profile.")
    parser.add_argument("--patience", type=int, help="Override patience from selected profile.")
    parser.add_argument("--num-workers", type=int, help="Override num_workers from selected profile.")
    parser.add_argument("--focal-alpha", type=float, help="Override focal_alpha from selected profile.")
    parser.add_argument("--total-epochs", type=int, default=100,
                        help="Total number of training epochs.")                                         # for setup total epochs  or can give directly epoch value here
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT,
                        help="Dataset root directory.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Cache directory for preprocessed samples.")
    parser.add_argument("--print-options", action="store_true",
                        help="Print GPU specs and HPARAM_TABLE then exit.")
    return parser.parse_args()


def print_gpu_info():
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        print("No CUDA GPU detected. Using CPU.")
        return
    print("\nDetected GPUs:")
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/(1024**3):.1f}GB")


def print_hparam_table():
    print("\nAvailable HPARAM_TABLE profiles:")
    print("  batch  | lr     | weight_decay | mixup_alpha | warmup_epochs | patience | num_workers | focal_alpha")
    print("  ------ | ------ | ------------ | ----------- | ------------- | -------- | ----------- | -----------")
    for batch, hp in sorted(HPARAM_TABLE.items()):
        print("  {:>5} | {:<6} | {:<12} | {:<11} | {:<13} | {:<8} | {:<11} | {:<11}".format(
            batch, hp["lr"], hp["weight_decay"], hp["mixup_alpha"],
            hp["warmup_epochs"], hp["patience"], hp["num_workers"], hp["focal_alpha"]))
    print("\nUse --batch-size to select a profile or pass individual overrides like --lr, --weight-decay, --num-workers.")


def choose_profile(default_batch):
    if not sys.stdin.isatty():
        return default_batch

    print("\nChoose a batch-size profile from HPARAM_TABLE, enter a custom batch size, or 'custom' for full customization.")
    print("Available profiles: 2, 4, 8. Custom sizes will use the closest profile.")
    print("Type 'custom' to set all hyperparameters manually.")

    while True:
        choice = input(f"Select batch size or 'custom' [auto]={default_batch}: ").strip().lower()
        if choice == "":
            return default_batch
        if choice in ("2", "4", "8"):
            return int(choice)
        if choice == "custom":
            return create_custom_profile()
        try:
            custom = int(choice)
            if custom <= 0:
                raise ValueError
            # Map to closest profile
            if custom < 3:
                return 2
            elif custom < 6:
                return 4
            else:
                return 8
        except ValueError:
            print("Invalid input. Enter 2, 4, 8, 'custom', or any positive integer for custom batch size.")


def create_custom_profile():
    """Prompt user for all hyperparameters to create a custom profile."""
    print("\nCreating custom profile. Enter values for each parameter (press Enter for defaults):")
    defaults = {
        "batch_size": 4,
        "lr": 3e-4,
        "weight_decay": 5e-4,
        "mixup_alpha": 0.3,
        "warmup_epochs": 2,
        "patience": 7,
        "num_workers": 2,
        "focal_alpha": 0.25,
    }
    custom_hp = {}
    for key, default in defaults.items():
        while True:
            try:
                value = input(f"  {key} [{default}]: ").strip()
                if value == "":
                    custom_hp[key] = default
                else:
                    if key in ["batch_size", "warmup_epochs", "patience", "num_workers"]:
                        custom_hp[key] = int(value)
                    else:
                        custom_hp[key] = float(value)
                break
            except ValueError:
                print(f"    Invalid value for {key}, try again.")
    custom_hp["note"] = "Custom_Profile"
    return custom_hp


def apply_interactive_overrides(hp, args):
    if not sys.stdin.isatty():
        return
    if any(getattr(args, opt) is not None for opt in [
            "weight_decay", "mixup_alpha", "lr", "warmup_epochs",
            "patience", "num_workers", "focal_alpha"]):
        return

    answer = input("Do you want to override any profile values? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        return

    overrides = [
        ("lr", float),
        ("weight_decay", float),
        ("mixup_alpha", float),
        ("warmup_epochs", int),
        ("patience", int),
        ("num_workers", int),
        ("focal_alpha", float),
    ]
    for name, cast in overrides:
        current = hp[name]
        value = input(f"  {name} [{current}]: ").strip()
        if value == "":
            continue
        try:
            hp[name] = cast(value)
        except ValueError:
            print(f"    Invalid value for {name}, keeping {current}.")


def override_hparams(hp, args):
    if args.weight_decay is not None:
        hp["weight_decay"] = args.weight_decay
    if args.mixup_alpha is not None:
        hp["mixup_alpha"] = args.mixup_alpha
    if args.lr is not None:
        hp["lr"] = args.lr
    if args.warmup_epochs is not None:
        hp["warmup_epochs"] = args.warmup_epochs
    if args.patience is not None:
        hp["patience"] = args.patience
    if args.num_workers is not None:
        hp["num_workers"] = args.num_workers
    if args.focal_alpha is not None:
        hp["focal_alpha"] = args.focal_alpha


def main():
    args = parse_args()

    print_gpu_info()
    print_hparam_table()
    if args.print_options:
        return

    default_batch = args.batch_size if args.batch_size is not None else (
        FORCE_BATCH_SIZE if FORCE_BATCH_SIZE is not None else detect_batch_size())
    selected = default_batch if args.batch_size is not None else choose_profile(default_batch)
    
    if isinstance(selected, dict):
        hp = selected
        batch_size = hp["batch_size"]
    else:
        batch_size = selected
        if batch_size not in HPARAM_TABLE:
            batch_size = 2
        hp = HPARAM_TABLE[batch_size].copy()
    
    override_hparams(hp, args)
    apply_interactive_overrides(hp, args)

    print(f"\nSelected profile: batch={batch_size}  [{hp['note']}]\n"
          f"  lr={hp['lr']}  wd={hp['weight_decay']}  "
          f"mixup={hp['mixup_alpha']}  focal_α={hp['focal_alpha']}  "
          f"warmup={hp['warmup_epochs']}  patience={hp['patience']}  "
          f"workers={hp['num_workers']}\n")

    root      = args.root
    cache_dir = args.cache_dir

    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)

    # ----------------------------------------------------------------
    # STEP 1: Build v2 cache using FFPPDatasetV2 directly
    # ----------------------------------------------------------------
    cache_builder = FFPPDatasetV2(
        video_paths=video_paths,
        labels=labels,             # label mean real or fake
        mask_paths=mask_paths,
        domains=domains,           # domain mean what type of fake (e.g. Deepfakes, Face2Face, etc.)
        cache_dir=cache_dir,
        use_alignment=True,
        training_mode=False,       # never augment during cache build
        label_smoothing=0.0,
        use_sbi=False,
        temporal_strategy="blend",
    )

    stats = cache_builder.get_cache_stats()
    print(f"Cache directory : {cache_dir}")
    print(f"Cache status    : {stats['cached']}/{stats['total']} cached, {stats['missing']} missing")

    if stats["missing"] > 0:
        print(f"\nBuilding {stats['missing']} missing cache entries (face detection + alignment).")
        print("This only runs once.")
        print("You can interrupt and restart; completed entries are not rebuilt.\n")
        cache_builder.init_face_detector()
        cache_builder.build_cache(num_workers=0)   # 0 workers = safe on Windows     but cpu has more core then and pu have more vram then try increase but for this i also don't know what it value should be  but i observe that in task manager performance section if number of worker more than required  then gou utilisation show traingular spike if it equla and less tahn required then gpu utilisation show stable flow
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

    TOTAL_EPOCHS = args.total_epochs

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
        batch_size=batch_size,
        num_workers=hp["num_workers"],
        lr=hp["lr"],
        backbone_lr_factor=0.1,
        weight_decay=hp["weight_decay"],
        patience=hp["patience"],
        warmup_epochs=hp["warmup_epochs"],
        use_amp=True,
        focal_alpha=hp["focal_alpha"],   # 0.25 — upweights real class
        focal_gamma=2.0,
        use_mixup=True,
        mixup_alpha=hp["mixup_alpha"],
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
