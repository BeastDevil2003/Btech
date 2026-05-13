import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_JET  = plt.cm.jet

_VIRIDIS = plt.cm.viridis

_COOL = plt.cm.cool


# ===========================================================================
# TENSOR → IMAGE CONVERSION
# ===========================================================================

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().float().numpy()
    img = img[:3]                          
    img = np.transpose(img, (1, 2, 0))    
    img = (img * 0.5 + 0.5) * 255.0       
    return np.clip(img, 0, 255).astype(np.uint8)


def mask_to_heatmap(mask: torch.Tensor, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    m = mask.squeeze().detach().cpu().float().numpy()
    m = (m * 255).astype(np.uint8)
    return cv2.applyColorMap(m, colormap)   


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1].copy()


def make_overlay(face_rgb: np.ndarray, mask_tensor: torch.Tensor, alpha=0.45) -> np.ndarray:
    heatmap_bgr = mask_to_heatmap(mask_tensor, cv2.COLORMAP_JET)
    heatmap_rgb = bgr_to_rgb(heatmap_bgr)

    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    overlay  = cv2.addWeighted(face_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return bgr_to_rgb(overlay)


def make_confidence_map(conf_tensor: torch.Tensor) -> np.ndarray:
    heatmap_bgr = mask_to_heatmap(conf_tensor, cv2.COLORMAP_VIRIDIS)
    return bgr_to_rgb(heatmap_bgr)


def make_stability_map(stability_tensor: torch.Tensor) -> np.ndarray:
    if stability_tensor.dim() == 5:
        stab = stability_tensor[0].mean(dim=0)           
    elif stability_tensor.dim() == 4:
        stab = stability_tensor[0].mean(dim=0, keepdim=True)
    else:
        stab = stability_tensor.unsqueeze(0)

    heatmap_bgr = mask_to_heatmap(stab, cv2.COLORMAP_COOL)
    return bgr_to_rgb(heatmap_bgr)

def _draw_colorbar_legend(
    ax,
    cmap,
    low_label: str,
    high_label: str,
    title: str,
    subtitle: str,
    bg_color: str = "white",
):
    ax.set_facecolor(bg_color)
    ax.axis("off")

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        extent=[0.05, 0.95, 0.25, 0.65],   
        transform=ax.transAxes,
    )

    rect = mpatches.FancyBboxPatch(
        (0.05, 0.25), 0.90, 0.40,
        boxstyle="square,pad=0",
        linewidth=0.5,
        edgecolor="gray",
        facecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(rect)

    ax.text(0.05, 0.18, low_label,  transform=ax.transAxes,
            fontsize=6.5, ha="left",  va="top", color="#444")
    ax.text(0.95, 0.18, high_label, transform=ax.transAxes,
            fontsize=6.5, ha="right", va="top", color="#444")

    ax.text(0.50, 0.98, title, transform=ax.transAxes,
            fontsize=7.5, ha="center", va="top", fontweight="bold", color="#111")

    ax.text(0.50, 0.16, subtitle, transform=ax.transAxes,
            fontsize=5.8, ha="center", va="top", color="#555",
            wrap=True, style="italic")


def _draw_face_legend(ax):
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(0.50, 0.85, "Original Face",
            transform=ax.transAxes, fontsize=7.5, ha="center", va="top",
            fontweight="bold", color="#111")
    ax.text(0.50, 0.55,
            "Middle frame from the\nvideo clip.\n"
            "Title color: green = correct\nprediction, red = wrong.",
            transform=ax.transAxes, fontsize=5.8, ha="center", va="top",
            color="#555", style="italic")


def _add_legend_row(fig, gs, legend_row_idx: int, n_cols: int):
    ax = fig.add_subplot(gs[legend_row_idx, :])
    ax.set_facecolor("#f8f8f8")
    ax.axis("off")

    legends = [
        dict(
            cmap=None,
            low="", high="",
            title="Row 1 — Original Face",
            subtitle=(
                "Middle frame of the video clip.  Title color: green = correct "
                "prediction, red = wrong prediction."
            ),
        ),
        dict(
            cmap=_JET,
            low="Real (blue)",
            high="Fake (red)",
            title="Row 2 — Forgery Mask Overlay  [JET]",
            subtitle=(
                "Red/Yellow = high forgery probability (most suspicious).  "
                "Blue = low probability (likely real).  "
                "Real faces: attention spreads over whole face.  "
                "Fake faces: tight rectangle = blending boundary artifact detected."
            ),
        ),
        dict(
            cmap=_VIRIDIS,
            low="Uncertain (dark)",
            high="Certain (yellow)",
            title="Row 3 — Confidence Map  [VIRIDIS]",
            subtitle=(
                "Yellow = model is very certain about its prediction here.  "
                "Dark purple = uncertain / low activation.  "
                "Yellow background = outside face mask (ignored region)."
            ),
        ),
        dict(
            cmap=_COOL,
            low="Stable (cyan)",
            high="Flickering (magenta)",
            title="Row 4 — Temporal Stability  [COOL]",
            subtitle=(
                "Cyan = region is consistent across frames (real indicator).  "
                "Magenta = region flickers between frames (fake indicator).  "
                "Solid magenta background = zero gradient / very certain region."
            ),
        ),
    ]

    box_w = 1.0 / len(legends)
    pad   = 0.01

    for k, leg in enumerate(legends):
        x0 = k * box_w + pad
        x1 = (k + 1) * box_w - pad

        # background box
        rect = mpatches.FancyBboxPatch(
            (x0, 0.04), x1 - x0, 0.92,
            boxstyle="round,pad=0.01",
            linewidth=0.5,
            edgecolor="#cccccc",
            facecolor="white",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

        # title
        ax.text(
            (x0 + x1) / 2, 0.91,
            leg["title"],
            transform=ax.transAxes,
            fontsize=6.2, ha="center", va="top",
            fontweight="bold", color="#111",
        )

        if leg["cmap"] is not None:
            # colorbar strip
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            cb_y0, cb_y1 = 0.48, 0.72
            ax.imshow(
                gradient,
                aspect="auto",
                cmap=leg["cmap"],
                extent=[x0 + 0.01, x1 - 0.01, cb_y0, cb_y1],
                transform=ax.transAxes,
                zorder=2,
            )
            border = mpatches.FancyBboxPatch(
                (x0 + 0.01, cb_y0), (x1 - x0 - 0.02), (cb_y1 - cb_y0),
                boxstyle="square,pad=0",
                linewidth=0.4,
                edgecolor="#999",
                facecolor="none",
                transform=ax.transAxes,
                zorder=3,
            )
            ax.add_patch(border)

            ax.text(x0 + 0.012, 0.44, leg["low"],
                    transform=ax.transAxes, fontsize=5.5,
                    ha="left", va="top", color="#444")
            ax.text(x1 - 0.012, 0.44, leg["high"],
                    transform=ax.transAxes, fontsize=5.5,
                    ha="right", va="top", color="#444")
        else:
            ax.text(
                (x0 + x1) / 2, 0.62,
                "[ photo ]",
                transform=ax.transAxes,
                fontsize=8, ha="center", va="center",
                color="#bbb", style="italic",
            )

        ax.text(
            (x0 + x1) / 2, 0.38,
            leg["subtitle"],
            transform=ax.transAxes,
            fontsize=5.2, ha="center", va="top",
            color="#555", style="italic",
            wrap=True,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# ===========================================================================
# BATCH VISUALIZATION GRID
# ===========================================================================

def visualize_batch(
    batch: dict,
    output: dict,
    save_path: str = "outputs",
    max_samples: int = 4,
    show_title: bool = True,
):
    os.makedirs(save_path, exist_ok=True)

    frames    = batch["frames"]    
    labels    = batch["label"]         
    pred_cls  = output["pred_cls"]     
    pred_mask = output["pred_mask"]      
    conf_map  = output["confidence"]  
    stability = output["stability"]    

    B = min(frames.shape[0], max_samples)
    N = frames.shape[1]
    mid_frame = N // 2

    probs = torch.sigmoid(pred_cls.detach().cpu()).squeeze(1)

    map_row_labels = ["Face", "Forgery Mask\n[JET]", "Confidence\n[VIRIDIS]", "Stability\n[COOL]"]
    n_map_rows  = len(map_row_labels)
    n_total_rows = n_map_rows + 1 

    row_heights = [4] * n_map_rows + [2.2]

    fig = plt.figure(figsize=(4 * B, sum(row_heights)))
    gs  = gridspec.GridSpec(
        n_total_rows, B,
        figure=fig,
        hspace=0.08,
        wspace=0.05,
        height_ratios=row_heights,
    )

    for i in range(B):
        face_tensor = frames[i, mid_frame]       
        face_rgb    = tensor_to_rgb(face_tensor)

        true_label = "REAL" if labels[i].item() < 0.5 else "FAKE"
        pred_prob  = probs[i].item()
        pred_label = "FAKE" if pred_prob > 0.5 else "REAL"
        title_color = "red" if true_label != pred_label else "green"

        images = [
            face_rgb,
            make_overlay(face_rgb, pred_mask[i]),
            make_confidence_map(conf_map[i]),
            make_stability_map(stability[i:i+1]),
        ]

        for row, img in enumerate(images):
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(img)
            ax.axis("off")

            if row == 0 and show_title:
                ax.set_title(
                    f"GT:{true_label}  Pred:{pred_label}({pred_prob:.2f})",
                    fontsize=8, color=title_color, pad=3,
                )

            if i == 0:
                ax.set_ylabel(
                    map_row_labels[row],
                    fontsize=8, rotation=90, labelpad=6,
                    va="center",
                )

    # -----------------------------------------------------------------------
    # Legend strip
    # -----------------------------------------------------------------------
    _add_legend_row(fig, gs, n_map_rows, B)

    plt.savefig(
        os.path.join(save_path, "visualization_grid.png"),
        dpi=150, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"Saved visualization grid → {save_path}/visualization_grid.png")


def visualize_sample(batch: dict, output: dict, save_path: str = "outputs"):

    os.makedirs(save_path, exist_ok=True)

    frames    = batch["frames"][0]        
    mid       = frames.shape[0] // 2
    face_rgb  = tensor_to_rgb(frames[mid])

    pred_mask = output["pred_mask"][0]    
    conf      = output["confidence"][0]  
    stability = output["stability"]       


    _save_with_legend(
        make_overlay(face_rgb, pred_mask),
        cmap=_JET,
        low_label="Real (blue)",
        high_label="Fake (red)",
        title="Forgery Mask Overlay  [JET colormap]",
        subtitle=(
            "Red/Yellow = high forgery probability  |  Blue = low probability (likely real)\n"
            "Real faces → diffuse attention.  Fake faces → tight rectangle = blending boundary."
        ),
        save_file=os.path.join(save_path, "overlay.png"),
    )

    _save_with_legend(
        make_confidence_map(conf),
        cmap=_VIRIDIS,
        low_label="Uncertain (dark purple)",
        high_label="Certain (yellow)",
        title="Confidence Map  [VIRIDIS colormap]",
        subtitle=(
            "Yellow = model is certain  |  Dark purple = uncertain / low activation\n"
            "Yellow background = outside face mask (ignored region)."
        ),
        save_file=os.path.join(save_path, "confidence.png"),
    )

    _save_with_legend(
        make_stability_map(stability),
        cmap=_COOL,
        low_label="Stable (cyan)",
        high_label="Flickering (magenta)",
        title="Temporal Stability  [COOL colormap]",
        subtitle=(
            "Cyan = consistent across frames (real indicator)  |  Magenta = flickering (fake indicator)\n"
            "Solid magenta background = zero gradient / model is very certain."
        ),
        save_file=os.path.join(save_path, "stability.png"),
    )

    if output.get("attribution") is not None:
        for key, val in output["attribution"].items():
            if val is None:
                continue
            fmap = val[0].mean(0).detach().cpu().float().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
            fmap = (fmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path, f"attribution_{key}.png"), heatmap)

    visualize_batch(batch, output, save_path=save_path)

    print(f"Saved all visualizations → {save_path}/")


def _save_with_legend(
    img_rgb: np.ndarray,
    cmap,
    low_label: str,
    high_label: str,
    title: str,
    subtitle: str,
    save_file: str,
):
    """
    Save img_rgb as a PNG with a horizontal colorbar legend strip below it.
    """
    h, w = img_rgb.shape[:2]
    fig_w = max(4.0, w / 100)
    legend_h = 1.1  

    fig = plt.figure(figsize=(fig_w, fig_w + legend_h))
    gs  = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[fig_w, legend_h],
        hspace=0.04,
    )

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_rgb)
    ax_img.axis("off")
    ax_img.set_title(title, fontsize=9, pad=4)

    ax_leg = fig.add_subplot(gs[1])
    ax_leg.set_facecolor("#f8f8f8")
    ax_leg.axis("off")

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_leg.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        extent=[0.05, 0.95, 0.45, 0.85],
        transform=ax_leg.transAxes,
    )
    border = mpatches.FancyBboxPatch(
        (0.05, 0.45), 0.90, 0.40,
        boxstyle="square,pad=0",
        linewidth=0.5, edgecolor="gray", facecolor="none",
        transform=ax_leg.transAxes,
    )
    ax_leg.add_patch(border)

    ax_leg.text(0.05, 0.40, low_label,  transform=ax_leg.transAxes,
                fontsize=7, ha="left",  va="top", color="#444")
    ax_leg.text(0.95, 0.40, high_label, transform=ax_leg.transAxes,
                fontsize=7, ha="right", va="top", color="#444")
    ax_leg.text(0.50, 0.30, subtitle,   transform=ax_leg.transAxes,
                fontsize=6.2, ha="center", va="top", color="#555",
                style="italic", multialignment="center")

    plt.savefig(save_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)