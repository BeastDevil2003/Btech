"""
  1. overall_metrics_bar.png     — overall bar chart (Acc/Prec/Rec/F1/AUC/IoU)
  2. domain_acc_bar.png          — ACC per manipulation method (like paper Fig 5)
  3. domain_auc_bar.png          — AUC per manipulation method (like paper Fig 6)
  4. domain_full_table.png       — all 6 metrics × all domains in one table image
  5. confusion_matrix.png        — overall confusion matrix
  6. roc_curve_overall.png       — overall ROC curve
  7. roc_curves_per_domain.png   — per-domain ROC curves overlaid on one plot
  8. domain_radar.png            — radar chart for multi-metric domain comparison
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)


DOMAIN_DISPLAY = {
    "Deepfakes":          "DF",
    "Face2Face":          "F2F",
    "FaceSwap":           "FS",
    "FaceShifter":        "FSh",
    "NeuralTextures":     "NT",
    "DeepFakeDetection":  "DFD",
    "youtube":            "Real-YT",
    "actors":             "Real-Act",
}

DOMAIN_COLORS = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706",
    "#7C3AED", "#0891B2", "#BE185D", "#065F46",
]

def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                threshold: float = 0.5) -> float:
    pred         = (pred_mask > threshold).float()
    gt           = (gt_mask   > 0.5).float()
    intersection = (pred * gt).sum()
    union        = pred.sum() + gt.sum() - intersection + 1e-6
    return float((intersection / union).item())


def compute_domain_metrics(labels, preds, probs, iou_list) -> dict:
    labels = np.asarray(labels, dtype=float)
    preds  = np.asarray(preds,  dtype=float)
    probs  = np.asarray(probs,  dtype=float)

    n       = len(labels)
    acc     = accuracy_score(labels, preds) if n > 0 else 0.0
    prec    = precision_score(labels, preds, zero_division=0)
    rec     = recall_score(labels, preds, zero_division=0)
    f1      = f1_score(labels, preds, zero_division=0)
    iou_val = float(np.mean(iou_list)) if iou_list else float("nan")

    n_classes = len(np.unique(labels))
    try:
        auc = roc_auc_score(labels, probs) if n_classes == 2 else float("nan")
    except Exception:
        auc = float("nan")

    return {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1":        f1,
        "AUC":       auc,
        "IoU":       iou_val,
        "N":         n,
    }

def evaluate(model, loader, device="cuda", return_details=False,
             threshold: float = 0.5):
    """
    metrics : dict   — overall {Accuracy, Precision, Recall, F1, AUC, IoU}
    details : dict   — raw arrays + per_domain dict (only when return_details=True)
    """
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    all_iou = []

    dom_labels  = defaultdict(list)
    dom_preds   = defaultdict(list)
    dom_probs   = defaultdict(list)
    dom_iou     = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            frames   = batch["frames"].to(device)
            labels   = batch["label"].to(device)
            masks    = batch["mask_frames"].to(device)
            has_mask = batch["has_mask"].to(device)
            domains  = batch["domain"]          

            out       = model(frames)
            pred_cls  = out["pred_cls"].squeeze(1)
            pred_mask = out["pred_mask"]

            probs_t = torch.sigmoid(pred_cls)
            probs_np = probs_t.cpu().numpy()
            preds_np = (probs_t >= threshold).float().cpu().numpy()
            labels_np = labels.cpu().numpy()

            mid = masks[:, masks.shape[1] // 2].unsqueeze(1)
            batch_iou = []
            for i in range(len(has_mask)):
                if has_mask[i]:
                    iou_val = compute_iou(pred_mask[i], mid[i])
                    all_iou.append(iou_val)
                    batch_iou.append((i, iou_val))

            all_labels.extend(labels_np.tolist())
            all_preds.extend(preds_np.tolist())
            all_probs.extend(probs_np.tolist())

            for i, dom in enumerate(domains):
                dom_labels[dom].append(float(labels_np[i]))
                dom_preds[dom].append(float(preds_np[i]))
                dom_probs[dom].append(float(probs_np[i]))

            for i, iou_val in batch_iou:
                dom = domains[i]
                dom_iou[dom].append(iou_val)

    all_labels_a = np.asarray(all_labels)
    all_preds_a  = np.asarray(all_preds)
    all_probs_a  = np.asarray(all_probs)

    overall = compute_domain_metrics(all_labels_a, all_preds_a, all_probs_a, all_iou)
    metrics = {k: v for k, v in overall.items() if k != "N"}

    if not return_details:
        return metrics
    real_domains = {"youtube", "actors"}
    real_lbls, real_prds, real_prbs = [], [], []
    for dom in dom_labels:
        if dom in real_domains:
            real_lbls.extend(dom_labels[dom])
            real_prds.extend(dom_preds[dom])
            real_prbs.extend(dom_probs[dom])
    real_lbls = np.asarray(real_lbls)
    real_prds = np.asarray(real_prds)
    real_prbs = np.asarray(real_prbs)

    per_domain = {}
    for dom in sorted(dom_labels.keys()):
        d_lbls = np.asarray(dom_labels[dom])
        d_prds = np.asarray(dom_preds[dom])
        d_prbs = np.asarray(dom_probs[dom])

        if dom in real_domains:
            per_domain[dom] = compute_domain_metrics(d_lbls, d_prds, d_prbs, dom_iou[dom])
        else:
            combined_lbls = np.concatenate([real_lbls, d_lbls])
            combined_prds = np.concatenate([real_prds, d_prds])
            combined_prbs = np.concatenate([real_prbs, d_prbs])
            combined_iou  = dom_iou[dom]
            m = compute_domain_metrics(combined_lbls, combined_prds,
                                        combined_prbs, combined_iou)
            m["N"] = len(d_lbls)
            per_domain[dom] = m

    return metrics, {
        "labels":     all_labels_a,
        "preds":      all_preds_a,
        "probs":      all_probs_a,
        "iou_scores": np.asarray(all_iou),
        "per_domain": per_domain,
        "dom_labels": dict(dom_labels),
        "dom_preds":  dict(dom_preds),
        "dom_probs":  dict(dom_probs),
    }


# =============================================================================
# CHART 1 — Overall metrics bar
# =============================================================================

def save_evaluation_record_image(metrics: dict, save_path: str,
                                  title: str = "Overall Metrics") -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names  = [k for k in metrics if k != "N"]
    values = [float(metrics[k]) for k in names]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [3, 2]})
    colors = ["#2563EB" if v >= 0.90 else "#D97706" if v >= 0.80 else "#DC2626"
              for v in values]
    bars = axes[0].bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_ylim(0.0, 1.12)
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_title(title, fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.015, f"{v*100:.2f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[1].axis("off")
    col1 = names[:3]; col2 = names[3:]
    v1   = [f"{metrics[k]*100:.2f}%" for k in col1]
    v2   = [f"{metrics[k]*100:.2f}%" for k in col2]
    text1 = "\n".join(f"{n:<14}: {v}" for n, v in zip(col1, v1))
    text2 = "\n".join(f"{n:<14}: {v}" for n, v in zip(col2, v2))
    axes[1].text(0.05, 0.90, text1, va="top", fontsize=12, family="monospace",
                 transform=axes[1].transAxes)
    axes[1].text(0.55, 0.90, text2, va="top", fontsize=12, family="monospace",
                 transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# CHART 2 — Per-domain ACC bar  
# =============================================================================

def save_domain_acc_bar(per_domain: dict, save_path: str,
                         model_name: str = "AFAGNetV3") -> str:
    """Bar chart of Accuracy per domain — matches Figure 5 in MH-FFNet paper."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    domains = sorted(per_domain.keys())
    labels  = [DOMAIN_DISPLAY.get(d, d) for d in domains]
    values  = [per_domain[d]["Accuracy"] * 100 for d in domains]
    colors  = DOMAIN_COLORS[:len(domains)]

    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.4), 5))
    bars = ax.bar(labels, values, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.8)

    min_v = max(0, min(values) - 5)
    ax.set_ylim(min_v, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Per-Manipulation Accuracy — {model_name} on FF++ (C23)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, (bar, d) in enumerate(zip(bars, domains)):
        n = per_domain[d]["N"]
        ax.text(bar.get_x() + bar.get_width() / 2,
                min_v + 0.5, f"n={n}",
                ha="center", va="bottom", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# CHART 3 — Per-domain AUC bar 
# =============================================================================

def save_domain_auc_bar(per_domain: dict, save_path: str,
                         model_name: str = "AFAGNetV3") -> str:
    """Bar chart of AUC per domain """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    domains = sorted(per_domain.keys())
    labels  = [DOMAIN_DISPLAY.get(d, d) for d in domains]
    values  = []
    for d in domains:
        v = per_domain[d]["AUC"]
        values.append(v * 100 if not math.isnan(v) else 0.0)
    colors = DOMAIN_COLORS[:len(domains)]

    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.4), 5))
    bars = ax.bar(labels, values, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.8)

    valid = [v for v in values if v > 0]
    min_v = max(0, min(valid) - 3) if valid else 0
    ax.set_ylim(min_v, 102)
    ax.set_ylabel("AUC (%)", fontsize=11)
    ax.set_title(f"Per-Manipulation AUC — {model_name} on FF++ (C23)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, v, d in zip(bars, values, domains):
        if math.isnan(per_domain[d]["AUC"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min_v + 1.5, "N/A\n(single class)",
                    ha="center", va="bottom", fontsize=7, color="gray")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# CHART 4 — Full domain metrics table image
# =============================================================================

def save_domain_table(per_domain: dict, overall: dict,
                       save_path: str, model_name: str = "AFAGNetV3") -> str:
    """
    Publication-quality table image: rows=domains, cols=metrics.
    Includes overall row at bottom. Colour-coded by value.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    metric_cols = ["N", "Accuracy", "Precision", "Recall", "F1", "AUC", "IoU"]
    domains_sorted = sorted(per_domain.keys())
    rows = domains_sorted + ["OVERALL"]

    data = []
    for d in domains_sorted:
        m = per_domain[d]
        row = [
            str(int(m["N"])),
            f"{m['Accuracy']*100:.2f}%",
            f"{m['Precision']*100:.2f}%",
            f"{m['Recall']*100:.2f}%",
            f"{m['F1']*100:.2f}%",
            f"{m['AUC']*100:.2f}%" if not math.isnan(m["AUC"]) else "—",
            f"{m['IoU']*100:.2f}%" if not math.isnan(m["IoU"]) else "—",
        ]
        data.append(row)
    data.append([
        str(sum(per_domain[d]["N"] for d in domains_sorted)),
        f"{overall.get('Accuracy',0)*100:.2f}%",
        f"{overall.get('Precision',0)*100:.2f}%",
        f"{overall.get('Recall',0)*100:.2f}%",
        f"{overall.get('F1',0)*100:.2f}%",
        f"{overall.get('AUC',0)*100:.2f}%",
        f"{overall.get('IoU',0)*100:.2f}%",
    ])

    n_rows = len(rows)
    n_cols = len(metric_cols)
    fig_h  = max(4, n_rows * 0.52 + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title(f"{model_name} — Per-Domain Evaluation on FF++ (C23)",
                 fontsize=13, fontweight="bold", pad=14)

    col_w = [0.06] + [0.13] * (n_cols - 1)   
    x_starts = [sum(col_w[:i]) for i in range(n_cols)]

    header_y = 0.97
    row_h    = (header_y - 0.05) / (n_rows + 1)

    for j, col in enumerate(metric_cols):
        ax.text(x_starts[j] + col_w[j] / 2, header_y,
                col, ha="center", va="top",
                fontsize=10, fontweight="bold",
                transform=ax.transAxes)
    ax.plot([0, 1], [header_y - 0.01, header_y - 0.01],
            color="#333333", linewidth=1.2, transform=ax.transAxes,
            clip_on=False, solid_capstyle="butt")

    for i, (row_label, row_data) in enumerate(zip(rows, data)):
        y = header_y - (i + 1) * row_h
        is_overall = (row_label == "OVERALL")
        bg_color   = "#E8F4FD" if is_overall else ("#F8F8F8" if i % 2 == 0 else "white")

        rect = FancyBboxPatch((0, y - row_h * 0.85), 1, row_h * 0.85,
                               boxstyle="square,pad=0",
                               linewidth=0, facecolor=bg_color,
                               transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

        display_label = DOMAIN_DISPLAY.get(row_label, row_label)
        ax.text(-0.01, y - row_h * 0.4,
                display_label,
                ha="right", va="center",
                fontsize=9, fontweight="bold" if is_overall else "normal",
                transform=ax.transAxes)

        for j, val in enumerate(row_data):
            cell_color = "black"
            if j > 0 and val != "—":
                try:
                    num = float(val.replace("%", ""))
                    if num >= 95:
                        cell_color = "#15803D"   # dark green
                    elif num >= 85:
                        cell_color = "#1D4ED8"   # blue
                    elif num >= 70:
                        cell_color = "#B45309"   # orange
                    else:
                        cell_color = "#DC2626"   # red
                except ValueError:
                    pass

            ax.text(x_starts[j] + col_w[j] / 2, y - row_h * 0.4,
                    val,
                    ha="center", va="center",
                    fontsize=9 if not is_overall else 10,
                    fontweight="bold" if is_overall else "normal",
                    color=cell_color,
                    transform=ax.transAxes)

    bottom_y = header_y - (n_rows + 0.5) * row_h
    ax.plot([0, 1], [bottom_y, bottom_y],
            color="#333333", linewidth=1.2, transform=ax.transAxes,
            clip_on=False, solid_capstyle="butt")

    for j in range(1, n_cols):
        x = x_starts[j] - 0.005
        ax.plot([x, x],
                [header_y - (n_rows + 0.55) * row_h, header_y],
                color="#CCCCCC", linewidth=0.7, transform=ax.transAxes,
                clip_on=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# CHART 5 — Confusion matrix
# =============================================================================

def save_confusion_matrix_image(labels, preds, save_path: str,
                                  title: str = "Confusion Matrix") -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"], fontsize=11)
    ax.set_yticklabels(["Real", "Fake"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# CHART 6 — Overall ROC curve
# =============================================================================

def save_roc_curve_image(labels, probs, save_path: str,
                          title: str = "ROC Curve") -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    labels, probs = np.asarray(labels), np.asarray(probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val     = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color="#2563EB", linewidth=2.2,
                label=f"AFAGNetV3 (AUC = {auc_val:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    except Exception:
        ax.text(0.5, 0.5, "ROC could not be computed", ha="center", va="center")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            linewidth=1.2, label="Random (AUC = 0.5000)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# CHART 7 — Per-domain ROC curves overlaid
# =============================================================================

def save_per_domain_roc(dom_labels: dict, dom_probs: dict,
                         save_path: str, model_name: str = "AFAGNetV3") -> str:
    """
    All per-domain ROC curves on one plot.
    Only domains with both real and fake samples can have a proper ROC.
    Single-class domains are skipped.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors    = DOMAIN_COLORS
    plotted   = 0
    for i, dom in enumerate(sorted(dom_labels.keys())):
        lbls  = np.asarray(dom_labels[dom])
        prbs  = np.asarray(dom_probs[dom])
        if len(np.unique(lbls)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(lbls, prbs)
            auc_val     = roc_auc_score(lbls, prbs)
            label       = f"{DOMAIN_DISPLAY.get(dom, dom)}  (AUC={auc_val:.4f})"
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    linewidth=1.8, label=label)
            plotted += 1
        except Exception:
            pass

    if plotted == 0:
        ax.text(0.5, 0.5, "No mixed-class domain found", ha="center", va="center")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Per-Domain ROC Curves — {model_name}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# CHART 8 — spider chart for multi-metric domain comparison
# =============================================================================

def save_domain_radar(per_domain: dict, save_path: str,
                       model_name: str = "AFAGNetV3") -> str:
    """
    Radar chart: each spoke = one metric, each polygon = one domain.
    Gives a quick visual of which domain is hard vs easy for the model.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    domains = sorted(per_domain.keys())
    N = len(metric_names)

    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]   

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax.grid(color="gray", alpha=0.3)

    for i, dom in enumerate(domains):
        m = per_domain[dom]
        values = []
        for mn in metric_names:
            v = m[mn]
            values.append(v if not math.isnan(v) else 0.0)
        values += values[:1]
        color = DOMAIN_COLORS[i % len(DOMAIN_COLORS)]
        ax.plot(angles, values, color=color, linewidth=1.8,
                label=DOMAIN_DISPLAY.get(dom, dom))
        ax.fill(angles, values, color=color, alpha=0.07)

    ax.set_title(f"Per-Domain Metric Radar — {model_name}",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# saves all charts at once
# =============================================================================

def save_evaluation_visuals(metrics: dict, labels, preds, probs,
                              save_dir: str = "eval",
                              prefix: str = "evaluation",
                              per_domain: dict = None,
                              dom_labels: dict = None,
                              dom_probs: dict = None,
                              model_name: str = "AFAGNetV3") -> dict:
    """
    Saves all evaluation charts.
    If per_domain / dom_labels / dom_probs are provided, domain charts are saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = {}

    saved["record"] = save_evaluation_record_image(
        metrics,
        os.path.join(save_dir, f"{prefix}_overall_metrics.png"),
        f"{model_name} — Overall Metrics on FF++ (C23)"
    )

    saved["confusion_matrix"] = save_confusion_matrix_image(
        labels, preds,
        os.path.join(save_dir, f"{prefix}_confusion_matrix.png"),
        f"{model_name} — Confusion Matrix"
    )

    saved["roc_curve"] = save_roc_curve_image(
        labels, probs,
        os.path.join(save_dir, f"{prefix}_roc_curve.png"),
        f"{model_name} — ROC Curve (FF++ C23)"
    )

    if per_domain:
        saved["domain_acc_bar"] = save_domain_acc_bar(
            per_domain,
            os.path.join(save_dir, f"{prefix}_domain_acc.png"),
            model_name
        )

        saved["domain_auc_bar"] = save_domain_auc_bar(
            per_domain,
            os.path.join(save_dir, f"{prefix}_domain_auc.png"),
            model_name
        )

        saved["domain_table"] = save_domain_table(
            per_domain, metrics,
            os.path.join(save_dir, f"{prefix}_domain_table.png"),
            model_name
        )

        saved["domain_radar"] = save_domain_radar(
            per_domain,
            os.path.join(save_dir, f"{prefix}_domain_radar.png"),
            model_name
        )

    if dom_labels and dom_probs:
        saved["per_domain_roc"] = save_per_domain_roc(
            dom_labels, dom_probs,
            os.path.join(save_dir, f"{prefix}_per_domain_roc.png"),
            model_name
        )

    return saved