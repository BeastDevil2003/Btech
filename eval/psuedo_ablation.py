import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from data.loader.splits import build_identity_disjoint_split
from models.afag_net_v3 import AFAGNetV3
from train.psuedo_mask import generate_pseudo_mask

ROOT      = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CANDIDATES = [
    os.path.join("checkpoints", "best_model.pth"),
    os.path.join("checkpoints", "latest_model.pth"),
]


# =============================================================================
# IoU HELPER
# =============================================================================

def pseudo_mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                    threshold: float = 0.5) -> float:
    pred         = (pred_mask > threshold).float()
    gt           = (gt_mask   > 0.5).float()
    intersection = (pred * gt).sum()
    union        = pred.sum() + gt.sum() - intersection + 1e-6
    return float((intersection / union).item())


# =============================================================================
# EVALUATE ONE WEIGHTING CONFIG
# =============================================================================

def evaluate_config(model, loader, alpha, beta, gamma, device=DEVICE):
    """
    Evaluate pseudo-mask quality for one (alpha, beta, gamma) combination.
    Computes IoU between generated pseudo-mask and ground-truth mask.
    Only evaluated on samples that HAVE ground-truth masks.

    alpha: GradCAM weight  (use_gradcam=False on 4GB — set True on T4)
    beta:  frequency branch weight
    gamma: noise branch weight
    """
    model.eval()
    scores = []

    # Disable GradCAM by default (safe for 4GB GPU)
    # On T4: change use_gradcam=True for better pseudo-mask quality
    use_gradcam = False

    for batch in loader:
        frames   = batch["frames"].to(device)
        masks    = batch["mask_frames"].to(device)
        has_mask = batch["has_mask"].to(device).bool()

        if not has_mask.any():
            continue

        with torch.enable_grad():
            out = model(frames)
            Fs  = out.get("Fs")
            Ff  = out.get("Ff")
            Fn  = out.get("Fn")

            if Fs is None or Ff is None or Fn is None:
                continue

            try:
                pseudo = generate_pseudo_mask(
                    model, frames, Fs, Ff, Fn,
                    alpha=alpha, beta=beta, gamma=gamma,
                    use_gradcam=use_gradcam,
                )
            except Exception as e:
                print(f"  [WARN] pseudo_mask failed: {e}")
                continue

        # Middle frame of pseudo and GT
        N = frames.shape[1]
        pseudo_mid = pseudo[:, N // 2]         # (B, 1, H, W)
        gt_mid     = masks[:, N // 2].unsqueeze(1)  # (B, 1, H, W)

        for i in range(len(has_mask)):
            if has_mask[i]:
                scores.append(pseudo_mask_iou(pseudo_mid[i], gt_mid[i]))

    return {
        "PseudoMaskIoU":    float(np.mean(scores)) if scores else 0.0,
        "SamplesWithMask":  len(scores),
    }


# =============================================================================
# RUN ALL CONFIGS
# =============================================================================

def run_pseudo_ablation(eval_dataset, model_path, device=DEVICE):
    """
    Test 6 weighting strategies for pseudo-mask generation.
    Results tell you which source of spatial signal is most informative.
    """
    configs = {
        "GradCAM+Freq+Noise (0.5/0.3/0.2)": (0.5, 0.3, 0.2),
        "GradCAM only       (1.0/0.0/0.0)": (1.0, 0.0, 0.0),
        "Frequency only     (0.0/1.0/0.0)": (0.0, 1.0, 0.0),
        "Noise only         (0.0/0.0/1.0)": (0.0, 0.0, 1.0),
        "GradCAM+Freq       (0.7/0.3/0.0)": (0.7, 0.3, 0.0),
        "GradCAM+Noise      (0.7/0.0/0.3)": (0.7, 0.0, 0.3),
        "Freq+Noise         (0.0/0.6/0.4)": (0.0, 0.6, 0.4),  # current default
    }

    model  = AFAGNetV3().to(device)
    state  = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    loader = DataLoader(eval_dataset, batch_size=2, shuffle=False, num_workers=0)

    results = {}
    print(f"\n{'='*65}")
    print(f"{'Config':<40} {'IoU':>8} {'N samples':>10}")
    print(f"{'─'*65}")

    for name, (a, b, c) in configs.items():
        m = evaluate_config(model, loader, a, b, c, device)
        results[name] = m
        print(f"{name:<40} {m['PseudoMaskIoU']:>8.4f} {m['SamplesWithMask']:>10}")

    print(f"{'='*65}")
    return results


# =============================================================================
# SAVE CHART
# =============================================================================

def save_ablation_chart(results: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names  = list(results.keys())
    values = [results[n]["PseudoMaskIoU"] for n in names]
    # Shorten names for chart
    short  = [n.split("(")[0].strip() for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2563EB" if v == max(values) else "#94A3B8" for v in values]
    bars   = ax.barh(short, values, color=colors, edgecolor="white")
    ax.set_xlabel("Pseudo-Mask IoU", fontsize=11)
    ax.set_title("Pseudo-Mask Ablation — IoU by Weight Configuration\n"
                 "(AFAGNetV3, FF++ C23)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(values) * 1.15)
    for bar, v in zip(bars, values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def resolve_model_path():
    for c in MODEL_CANDIDATES:
        if not os.path.exists(c):
            continue
        sd = torch.load(c, map_location="cpu")
        if all(torch.isfinite(v).all() for v in sd.values()):
            return c
    raise FileNotFoundError("No healthy checkpoint found.")


def main():
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(ROOT)

    _, val_indices = build_identity_disjoint_split(
        video_paths, val_ratio=0.2, seed=42  # MUST match training split
    )

    dataset = FFPPDatasetV2(
        video_paths=video_paths, labels=labels,
        mask_paths=mask_paths, domains=domains,
        cache_dir=CACHE_DIR, use_alignment=True,
        training_mode=False, label_smoothing=0.0,
        use_sbi=False,
    )
    eval_dataset = Subset(dataset, val_indices)
    model_path   = resolve_model_path()

    print(f"Checkpoint : {model_path}")
    print(f"Eval size  : {len(eval_dataset)}")
    print(f"Note: GradCAM disabled (use_gradcam=False) — safe for 4GB GPU")
    print(f"      On T4: change use_gradcam=True in evaluate_config()")

    results = run_pseudo_ablation(eval_dataset, model_path, DEVICE)

    save_path = os.path.join("eval", "pseudo_ablation.png")
    save_ablation_chart(results, save_path)


if __name__ == "__main__":
    main()
