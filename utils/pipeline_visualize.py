import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.afag_net_v3 import AFAGNetV3
from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from data.loader.splits import build_identity_disjoint_split
from torch.utils.data import Subset

ROOT      = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"



def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) in [-1,1] → (H,W,3) uint8 RGB.  Handles 3-ch and 6-ch."""
    img = t.detach().cpu().float().numpy()[:3]
    img = np.transpose(img, (1, 2, 0))
    img = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return img


def feature_map_to_rgb(feat: torch.Tensor,
                        target_hw: tuple = (224, 224),
                        colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:

    feat = feat.detach().cpu().float()
    energy = feat.norm(dim=0)
    mn, mx = energy.min(), energy.max()
    if mx - mn > 1e-6:
        energy = (energy - mn) / (mx - mn)
    else:
        energy = torch.zeros_like(energy)
    energy_np = (energy.numpy() * 255).astype(np.uint8)
    resized = cv2.resize(energy_np, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_NEAREST)
    bgr = cv2.applyColorMap(resized, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def similarity_map_to_rgb(sim: torch.Tensor,
                           target_hw: tuple = (224, 224)) -> np.ndarray:
    sim = sim.squeeze().detach().cpu().float().numpy()
    sim = np.clip(sim, -1, 1)
    sim_01 = (sim + 1) / 2.0         
    sim_u8 = (sim_01 * 255).astype(np.uint8)
    resized = cv2.resize(sim_u8, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_NEAREST)
    bgr = cv2.applyColorMap(resized, cv2.COLORMAP_RdYlGn if hasattr(cv2, 'COLORMAP_RdYlGn')
                            else cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_to_rgb(mask: torch.Tensor,
                target_hw: tuple = (224, 224),
                colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """(1,H,W) mask in [0,1] → RGB heatmap."""
    m = mask.squeeze().detach().cpu().float().numpy()
    m = np.clip(m, 0, 1)
    m_u8 = (m * 255).astype(np.uint8)
    if m_u8.shape != (target_hw[0], target_hw[1]):
        m_u8 = cv2.resize(m_u8, (target_hw[1], target_hw[0]))
    bgr = cv2.applyColorMap(m_u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(face_rgb: np.ndarray, heatmap_rgb: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over a face image."""
    h, w = face_rgb.shape[:2]
    hmap = cv2.resize(heatmap_rgb, (w, h))
    return np.clip(
        (1 - alpha) * face_rgb.astype(float) + alpha * hmap.astype(float),
        0, 255
    ).astype(np.uint8)


def add_label(ax, text: str, color: str = "white", fontsize: int = 8):
    """Add a label inside the axes (top-left corner)."""
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            fontsize=fontsize, color=color,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.55))


def colorbar_axes(fig, ax, label_lo: str, label_hi: str,
                  colormap=cv2.COLORMAP_VIRIDIS):
    """Add a tiny horizontal colorbar below an axes."""
    gradient = np.linspace(0, 255, 256, dtype=np.uint8).reshape(1, 256)
    bgr_strip = cv2.applyColorMap(gradient, colormap)
    rgb_strip  = cv2.cvtColor(bgr_strip, cv2.COLOR_BGR2RGB)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.025, pos.width, 0.012])
    cax.imshow(rgb_strip, aspect='auto')
    cax.set_xticks([0, 255])
    cax.set_xticklabels([label_lo, label_hi], fontsize=6)
    cax.set_yticks([])


# =============================================================================
# CHANNEL-WISE TOP-K FEATURE MAPS
# =============================================================================

def top_k_channels(feat: torch.Tensor, k: int = 4,
                   target_hw: tuple = (112, 112)) -> list:
    """
    Return the top-k most active channel feature maps from (C,H,W).
    'Most active' = highest mean absolute value.
    Returns list of k (H,W) float arrays in [0,1].
    """
    feat  = feat.detach().cpu().float()   
    score = feat.abs().mean(dim=(1, 2))   
    topk  = score.topk(min(k, feat.shape[0])).indices.tolist()
    maps  = []
    for idx in topk:
        m = feat[idx].numpy()
        mn, mx = m.min(), m.max()
        m = (m - mn) / (mx - mn + 1e-6)
        m = cv2.resize(m.astype(np.float32), (target_hw[1], target_hw[0]))
        maps.append(m)
    return maps


# =============================================================================
# MAIN PIPELINE VISUALIZATION
# =============================================================================

def visualize_pipeline(sample: dict, output: dict,
                       save_dir: str = "outputs/pipeline",
                       sample_idx: int = 0):

    os.makedirs(save_dir, exist_ok=True)

    frames    = sample["frames"]        
    label     = sample["label"].item()   
    N         = frames.shape[0]
    mid       = N // 2

    face_rgb  = tensor_to_rgb(frames[mid])  


    F_low     = output["F_low"][0]           # (256,14,14)
    F_high    = output["F_high"][0]          # (256, 7, 7)
    Fs        = output["Fs"][0]              # (256,14,14)
    Ff        = output["Ff"][0]              # (256,14,14)
    Fn        = output["Fn"][0]              # (256,14,14)
    Ffusion   = output["Ffusion"][0]         # (256,14,14)
    Csf       = output["Csf"][0]             # (1,14,14) or (1,H,W)
    Csn       = output["Csn"][0]             # (1,14,14)
    pred_mask = output["pred_mask"][0]       # (1,224,224)
    conf_map  = output["confidence"][0]      # (1,224,224)
    stability = output["stability"]          # (B,N-1,1,H,W)
    pred_cls  = output["pred_cls"][0]        # (1,)

    prob        = torch.sigmoid(pred_cls).item()
    true_label  = "REAL" if label < 0.5 else "FAKE"
    pred_label  = "FAKE" if prob > 0.5 else "REAL"
    correct     = (true_label == pred_label)
    title_color = "#00C851" if correct else "#FF4444"

    stab = stability[0].mean(dim=0)      
    if stab.dim() == 2:
        stab = stab.unsqueeze(0)

    HW = (224, 224)

    hm = {
        "face":      face_rgb,
        "f_low":     feature_map_to_rgb(F_low,   HW, cv2.COLORMAP_PLASMA),
        "f_high":    feature_map_to_rgb(F_high,  HW, cv2.COLORMAP_PLASMA),
        "Fs":        feature_map_to_rgb(Fs,      HW, cv2.COLORMAP_HOT),
        "Ff":        feature_map_to_rgb(Ff,      HW, cv2.COLORMAP_OCEAN),
        "Fn":        feature_map_to_rgb(Fn,      HW, cv2.COLORMAP_SPRING),
        "Csf":       similarity_map_to_rgb(Csf,  HW),
        "Csn":       similarity_map_to_rgb(Csn,  HW),
        "Ffusion":   feature_map_to_rgb(Ffusion, HW, cv2.COLORMAP_TURBO),
        "mask_ov":   overlay_heatmap(face_rgb, mask_to_rgb(pred_mask, HW, cv2.COLORMAP_JET)),
        "conf":      mask_to_rgb(conf_map, HW, cv2.COLORMAP_VIRIDIS),
        "stability": mask_to_rgb(stab,     HW, cv2.COLORMAP_COOL),
    }


    fig1 = plt.figure(figsize=(22, 24))
    fig1.patch.set_facecolor("#0D0D0D")

    fig1.suptitle(
        f"AFAGNetV3 — Full Pipeline Feature Visualization\n"
        f"GT: {true_label}   Predicted: {pred_label} ({prob:.3f})   "
        f"{'✓ Correct' if correct else '✗ Wrong'}",
        fontsize=14, color=title_color, fontweight="bold", y=0.98
    )

    gs1 = gridspec.GridSpec(4, 4, figure=fig1,
                            hspace=0.38, wspace=0.08,
                            left=0.03, right=0.97, top=0.94, bottom=0.02)

    def _show(ax, img, title, label_text=None, cmap=None):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=8.5, color="white", pad=3, fontweight="bold")
        ax.axis("off")
        if label_text:
            add_label(ax, label_text, color="cyan", fontsize=7)

    _show(fig1.add_subplot(gs1[0, 0]), hm["face"],
          "Step 0 · Input Frame (middle)", "RGB face crop\n(224×224, 6-ch input)")

    _show(fig1.add_subplot(gs1[0, 1]), hm["f_low"],
          "Step 1a · Backbone F_low (28×28)",
          "Local texture features\nMobileViT stage-3, 256ch\n(item #13: was 14×14)")

    _show(fig1.add_subplot(gs1[0, 2]), hm["f_high"],
          "Step 1b · Backbone F_high (7×7)",
          "Semantic context features\nMobileViT stage-5, 256ch\n(upsampled for display)")

    f_low_ov = overlay_heatmap(face_rgb, hm["f_low"], alpha=0.5)
    _show(fig1.add_subplot(gs1[0, 3]), f_low_ov,
          "Step 1 · F_low overlay on face",
          "Where backbone attends\non the raw face pixels")

    _show(fig1.add_subplot(gs1[1, 0]), hm["Fs"],
          "Step 2a · Spatial Branch Fs",
          "Dual-path CBAM attention\nF_low + F_high fusion\nHot = high spatial activation")

    _show(fig1.add_subplot(gs1[1, 1]), hm["Ff"],
          "Step 2b · Frequency Branch Ff",
          "FFT magnitude + phase\nGlobal & 2×2 block FFT\nOcean = freq artifact energy")

    _show(fig1.add_subplot(gs1[1, 2]), hm["Fn"],
          "Step 2c · Noise Branch Fn",
          "SRM noise residual\nLearnable high-pass filter\nSpring = noise pattern energy")

    comparison = np.hstack([
        cv2.resize(hm["Fs"], (74, 224)),
        cv2.resize(hm["Ff"], (74, 224)),
        cv2.resize(hm["Fn"], (74, 224)),
    ])
    ax_cmp = fig1.add_subplot(gs1[1, 3])
    ax_cmp.imshow(comparison)
    ax_cmp.set_title("Branch energy comparison\nFs | Ff | Fn (side by side)",
                     fontsize=8.5, color="white", pad=3, fontweight="bold")
    ax_cmp.axis("off")
    ax_cmp.axvline(x=74,  color="white", linewidth=1.0, alpha=0.6)
    ax_cmp.axvline(x=148, color="white", linewidth=1.0, alpha=0.6)
    for x, label in zip([37, 111, 185], ["Spatial", "Freq", "Noise"]):
        ax_cmp.text(x, 215, label, ha="center", fontsize=7, color="white")

    _show(fig1.add_subplot(gs1[2, 0]), hm["Csf"],
          "Step 3a · CGAF Similarity Csf",
          "Spatial–Frequency agreement\nRed=agree (+1), Blue=disagree\nHigh = both detect same region")

    _show(fig1.add_subplot(gs1[2, 1]), hm["Csn"],
          "Step 3b · CGAF Similarity Csn",
          "Spatial–Noise agreement\nRed=agree, Blue=disagree\nConsistency across branches")

    cfn_approx = overlay_heatmap(face_rgb,
                                 overlay_heatmap(hm["Csf"], hm["Csn"], alpha=0.5),
                                 alpha=0.4)
    _show(fig1.add_subplot(gs1[2, 2]), cfn_approx,
          "Step 3c · Cross-branch context",
          "Combined Csf+Csn signal\noverlaid on face\n(blended for display)")

    _show(fig1.add_subplot(gs1[2, 3]), hm["Ffusion"],
          "Step 3d · Fused Features Ffusion",
          "CGAFv2 gated adaptive fusion\nWeighted Fs + Ff + Fn\nTurbo = fused activation")

    _show(fig1.add_subplot(gs1[3, 0]), hm["mask_ov"],
          "Step 4a · Forgery Mask (overlay)",
          "LocalizationHead output\nRed/Yellow = high forgery\nBlue = likely real region")

    _show(fig1.add_subplot(gs1[3, 1]), hm["conf"],
          "Step 4b · Confidence Map",
          "Pixel-level certainty\n|mask - 0.5| × 2\nYellow = very certain")

    _show(fig1.add_subplot(gs1[3, 2]), hm["stability"],
          "Step 4c · Temporal Stability",
          "Frame-to-frame consistency\nMagenta = flickering (fake)\nCyan = stable (real)")

    ax_pred = fig1.add_subplot(gs1[3, 3])
    ax_pred.set_facecolor("#111111")
    bar_color = "#FF4444" if pred_label == "FAKE" else "#00C851"
    ax_pred.barh(0.5, prob, height=0.3, color=bar_color, alpha=0.85)
    ax_pred.barh(0.5, 1.0, height=0.3, color="#333333", alpha=0.4)
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(0, 1)
    ax_pred.axvline(x=0.5, color="white", linestyle="--", linewidth=1, alpha=0.6)
    ax_pred.set_title("Step 4d · Final Prediction",
                      fontsize=8.5, color="white", pad=3, fontweight="bold")
    ax_pred.text(0.5, 0.82, f"GT: {true_label}",
                 ha="center", va="center", fontsize=11, color="white",
                 transform=ax_pred.transAxes)
    ax_pred.text(0.5, 0.60, f"Pred: {pred_label}",
                 ha="center", va="center", fontsize=13, color=bar_color,
                 fontweight="bold", transform=ax_pred.transAxes)
    ax_pred.text(0.5, 0.38, f"P(fake) = {prob:.4f}",
                 ha="center", va="center", fontsize=10, color="white",
                 transform=ax_pred.transAxes)
    verdict = "✓ Correct" if correct else "✗ Wrong"
    ax_pred.text(0.5, 0.18, verdict,
                 ha="center", va="center", fontsize=12,
                 color="#00C851" if correct else "#FF4444",
                 fontweight="bold", transform=ax_pred.transAxes)
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_edgecolor("#555555")

    path1 = os.path.join(save_dir, f"pipeline_overview_{sample_idx}.png")
    fig1.savefig(path1, dpi=130, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close(fig1)
    print(f"  Saved: {path1}")


    branches = {
        "Backbone F_low\n(texture detector)":  F_low,
        "Spatial Branch Fs\n(CBAM attention)":  Fs,
        "Frequency Branch Ff\n(FFT artifacts)": Ff,
        "Noise Branch Fn\n(SRM residual)":      Fn,
        "Fused Ffusion\n(CGAFv2 output)":       Ffusion,
    }
    n_branches = len(branches)
    k = 4   # top-k channels to show

    fig2 = plt.figure(figsize=(k * 3, n_branches * 3.2))
    fig2.patch.set_facecolor("#0D0D0D")
    fig2.suptitle("Top-4 Most Activated Channels per Branch\n"
                  "Each cell = single feature channel (highest mean activation)",
                  fontsize=12, color="white", fontweight="bold", y=0.99)

    gs2 = gridspec.GridSpec(n_branches, k, figure=fig2,
                            hspace=0.4, wspace=0.05,
                            left=0.08, right=0.98, top=0.94, bottom=0.02)

    cmaps = [cv2.COLORMAP_PLASMA, cv2.COLORMAP_HOT,
             cv2.COLORMAP_OCEAN, cv2.COLORMAP_SPRING, cv2.COLORMAP_TURBO]

    for row, (branch_name, feat_tensor) in enumerate(branches.items()):
        channel_maps = top_k_channels(feat_tensor, k=k, target_hw=(112, 112))
        for col, ch_map in enumerate(channel_maps):
            ax = fig2.add_subplot(gs2[row, col])
            ch_bgr = cv2.applyColorMap((ch_map * 255).astype(np.uint8),
                                       cmaps[row % len(cmaps)])
            ch_rgb = cv2.cvtColor(ch_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(ch_rgb)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(branch_name, fontsize=8, color="white",
                              rotation=90, labelpad=5)
            ax.set_title(f"ch#{col+1}", fontsize=7, color="#AAAAAA", pad=2)

    path2 = os.path.join(save_dir, f"branch_channels_{sample_idx}.png")
    fig2.savefig(path2, dpi=130, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    n_show = min(N, 8)
    step   = max(N // n_show, 1)
    sel_frames = list(range(0, N, step))[:n_show]

    fig3 = plt.figure(figsize=(n_show * 2.5, 6))
    fig3.patch.set_facecolor("#0D0D0D")
    fig3.suptitle(f"Temporal Strip — {n_show} of {N} frames  "
                  f"(GT:{true_label}, Pred:{pred_label} p={prob:.3f})",
                  fontsize=11, color="white", fontweight="bold")

    gs3 = gridspec.GridSpec(2, n_show, figure=fig3,
                            hspace=0.1, wspace=0.04,
                            left=0.02, right=0.98, top=0.88, bottom=0.05)

    mask_seq = output.get("mask_sequence")  
    if mask_seq is not None:
        mask_seq = mask_seq[0]              

    for col, fi in enumerate(sel_frames):
        ax_f = fig3.add_subplot(gs3[0, col])
        ax_f.imshow(tensor_to_rgb(frames[fi]))
        ax_f.axis("off")
        if col == 0:
            ax_f.set_ylabel("Input", fontsize=7, color="white")
        ax_f.set_title(f"f{fi}", fontsize=7, color="#AAAAAA", pad=2)

        ax_m = fig3.add_subplot(gs3[1, col])
        if mask_seq is not None:
            m_rgb = mask_to_rgb(mask_seq[fi], (224, 224), cv2.COLORMAP_JET)
            ax_m.imshow(overlay_heatmap(tensor_to_rgb(frames[fi]), m_rgb, alpha=0.5))
        else:
            ax_m.imshow(hm["mask_ov"])
        ax_m.axis("off")
        if col == 0:
            ax_m.set_ylabel("Mask", fontsize=7, color="white")

    path3 = os.path.join(save_dir, f"temporal_strip_{sample_idx}.png")
    fig3.savefig(path3, dpi=130, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close(fig3)
    print(f"  Saved: {path3}")

    return path1, path2, path3



def resolve_model(checkpoint=None):
    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        sd = torch.load(checkpoint, map_location="cpu")
        if all(torch.isfinite(v).all() for v in sd.values()):
            return checkpoint, sd
        raise RuntimeError(f"Checkpoint {checkpoint} contains invalid values.")

    for c in ["checkpoints/best_model.pth", "checkpoints/ema_model.pth"]:
        if not os.path.exists(c):
            continue
        sd = torch.load(c, map_location="cpu")
        if all(torch.isfinite(v).all() for v in sd.values()):
            return c, sd
    raise FileNotFoundError("No healthy checkpoint. Train first.")


def build_single_video_dataset(video_path, label, domain, mask_path):
    if label is None:
        label = 0
    return [video_path], [label], [mask_path], [domain]


def extract_state_dict(loaded):
    if isinstance(loaded, dict):
        if "state_dict" in loaded:
            return loaded["state_dict"]
        if "model_state_dict" in loaded:
            return loaded["model_state_dict"]
    return loaded


def load_state_dict_safe(model, state_dict):
    model_dict = model.state_dict()
    compatible = {}
    skipped = []
    for name, param in state_dict.items():
        if name not in model_dict:
            skipped.append(name)
            continue
        if model_dict[name].shape != param.shape:
            skipped.append(name)
            continue
        compatible[name] = param

    if not compatible:
        raise RuntimeError(
            "No compatible weights were found in the checkpoint for the current model architecture. "
            "Please use a checkpoint that matches the current AFAGNetV3 version."
        )

    model.load_state_dict(compatible, strict=False)
    missing = [name for name in model_dict if name not in compatible]
    return compatible, skipped, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",   default="outputs/pipeline",
                        help="Directory to save visualizations")
    parser.add_argument("--n_samples",  type=int, default=4,
                        help="Number of samples to visualize (2 real + 2 fake per pair)")
    parser.add_argument("--video", type=str, default=None,
                        help="Optional direct video path for single-sample visualization.")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None,
                        help="Optional label for the provided video (0=real, 1=fake).")
    parser.add_argument("--domain", type=str, default="youtube",
                        help="Domain name for the provided video.")
    parser.add_argument("--mask", type=str, default=None,
                        help="Optional mask video path for a fake video.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint path to load instead of the default candidates.")
    parser.add_argument("--video_idx",  type=int, default=None,
                        help="Force a specific dataset index (overrides n_samples)")
    args = parser.parse_args()

    # Load model
    model_path, sd = resolve_model(args.checkpoint)
    print(f"Loading: {model_path}")
    model = AFAGNetV3().to(DEVICE)
    state_dict = extract_state_dict(sd)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("WARNING: strict state_dict load failed. Trying compatible subset...")
        state_dict = extract_state_dict(sd)
        compatible, skipped, missing = load_state_dict_safe(model, state_dict)
        print(f"  Loaded {len(compatible)}/{len(model.state_dict())} compatible parameters.")
        print(f"  Skipped {len(skipped)} checkpoint keys due to mismatch or unexpected names.")
        print(f"  {len(missing)} model parameters were not initialized from checkpoint.")
    model.eval()

    if args.video is not None:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")
        if args.mask is not None and not os.path.exists(args.mask):
            raise FileNotFoundError(f"Mask video not found: {args.mask}")

        video_paths, labels, mask_paths, domains = build_single_video_dataset(
            args.video, args.label, args.domain, args.mask
        )
        dataset = FFPPDatasetV2(
            video_paths=video_paths, labels=labels,
            mask_paths=mask_paths, domains=domains,
            cache_dir=CACHE_DIR, use_alignment=True,
            training_mode=False, label_smoothing=0.0,
            use_sbi=False,
        )
        indices = [0]
    else:
        video_paths, labels, mask_paths, domains = build_ffpp_dataset(ROOT)
        dataset = FFPPDatasetV2(
            video_paths=video_paths, labels=labels,
            mask_paths=mask_paths, domains=domains,
            cache_dir=CACHE_DIR, use_alignment=True,
            training_mode=False, label_smoothing=0.0,
            use_sbi=False,
        )

        if args.video_idx is not None:
            indices = [args.video_idx]
        else:
            n_each  = max(args.n_samples // 2, 1)
            _, val_idx = build_identity_disjoint_split(video_paths, val_ratio=0.2, seed=42)
            val_set = set(val_idx)
            real_pool = [i for i in val_idx if labels[i] == 0][:n_each]
            fake_pool = [i for i in val_idx if labels[i] == 1][:n_each]
            indices   = real_pool + fake_pool

    print(f"Visualizing {len(indices)} samples → {args.save_dir}/")

    for viz_num, idx in enumerate(indices):
        sample = dataset[idx]
        frames = sample["frames"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(frames)

        output_single = {}
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                output_single[k] = v.cpu()
            else:
                output_single[k] = v

        sample_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in sample.items()}

        domain = sample["domain"]
        lbl    = "REAL" if sample["label"].item() < 0.5 else "FAKE"
        print(f"\nSample {viz_num+1}/{len(indices)}: idx={idx}  domain={domain}  GT={lbl}")

        sub_dir = os.path.join(args.save_dir, f"sample_{viz_num:02d}_{lbl}_{domain}")
        visualize_pipeline(sample_cpu, output_single,
                           save_dir=sub_dir, sample_idx=viz_num)

    print(f"\nDone. All visualizations saved in: {args.save_dir}/")
    print("Each sample folder contains:")
    print("  pipeline_overview_N.png  — full 4-row pipeline (input → prediction)")
    print("  branch_channels_N.png    — top-4 activated channels per branch")
    print("  temporal_strip_N.png     — N-frame mask evolution over time")


if __name__ == "__main__":
    main()