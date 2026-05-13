import os, math, cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset, Dataset
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from data.loader.build_dataset   import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from data.loader.splits          import build_identity_disjoint_split
from eval.evaluate               import evaluate
from models.afag_net_v3          import AFAGNetV3

ROOT      = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CANDIDATES = ["checkpoints/best_model.pth", "checkpoints/ema_model.pth"]

# Exact params from MH-FFNet Table 3
ATTACK_PARAMS = {
    "CS":   [None, 0.4,  0.3,   0.2,   0.1,  0.0],
    "CC":   [None, 0.85, 0.725, 0.6,   0.475,0.35],
    "BW":   [None, 16,   32,    48,    64,   80],
    "GN":   [None, 0.001,0.002, 0.005, 0.01, 0.05],
    "GB":   [None, 3,    5,     7,     9,    13],
    "JPEG": [None, 90,   70,    50,    30,   20],
    "RO":   [None, 30,   60,    90,    120,  150],
    "AF":   [None, 0.8,  0.9,   1.0,   1.1,  1.2],
}
ATTACK_NAMES = {
    "CS":"Color Saturation","CC":"Color Contrast","BW":"Block-wise",
    "GN":"Gaussian Noise","GB":"Gaussian Blur","JPEG":"JPEG Compression",
    "RO":"Rotate","AF":"Affine Transform",
}
# Published Table 4 results
PAPER_RESULTS = {
    "SPSL":    {"CS":98.41,"CC":96.86,"BW":84.65,"GN":55.63,"GB":91.03,"JPEG":89.01,"RO":65.04,"AF":96.10},
    "MAT":     {"CS":98.28,"CC":97.97,"BW":91.97,"GN":67.66,"GB":96.63,"JPEG":91.57,"RO":85.19,"AF":97.86},
    "GocNet":  {"CS":94.30,"CC":94.36,"BW":68.94,"GN":47.10,"GB":90.27,"JPEG":84.53,"RO":64.22,"AF":94.73},
    "HIFE":    {"CS":97.07,"CC":96.19,"BW":88.74,"GN":56.95,"GB":94.07,"JPEG":88.48,"RO":76.36,"AF":97.74},
    "MH-FFNet":{"CS":98.42,"CC":98.59,"BW":94.69,"GN":77.84,"GB":97.66,"JPEG":91.04,"RO":84.10,"AF":98.73},
}


def apply_attack(frames, attack, severity):
    if severity == 0:
        return frames
    param = ATTACK_PARAMS[attack][severity]
    N, C, H, W = frames.shape

    if attack == "CS":
        result = []
        for n in range(N):
            img = TF.to_pil_image(torch.clamp((frames[n,:3]+1)/2, 0, 1))
            img = ImageEnhance.Color(img).enhance(param)
            t = TF.to_tensor(img)*2-1
            result.append(torch.cat([t, frames[n,3:]], 0) if C==6 else t)
        return torch.stack(result)

    elif attack == "CC":
        result = []
        for n in range(N):
            img = TF.to_pil_image(torch.clamp((frames[n,:3]+1)/2, 0, 1))
            img = ImageEnhance.Contrast(img).enhance(param)
            t = TF.to_tensor(img)*2-1
            result.append(torch.cat([t, frames[n,3:]], 0) if C==6 else t)
        return torch.stack(result)

    elif attack == "BW":
        out = frames.clone()
        bh  = max(1, H//14); bw = max(1, W//14)
        for _ in range(int(param)):
            r = np.random.randint(0, H-bh+1)
            c = np.random.randint(0, W-bw+1)
            out[:,:3,r:r+bh,c:c+bw] = float(np.random.uniform(-1,1))
        return out

    elif attack == "GN":
        return torch.clamp(frames + torch.randn_like(frames)*float(np.sqrt(param)), -1, 1)

    elif attack == "GB":
        k = int(param)
        if k < 3: return frames
        result = []
        for n in range(N):
            img = ((frames[n,:3]+1)/2*255).byte().permute(1,2,0).cpu().numpy()
            img = cv2.GaussianBlur(img, (k,k), 0)
            t = torch.from_numpy(img).permute(2,0,1).float()/255*2-1
            result.append(torch.cat([t, frames[n,3:].cpu()], 0) if C==6 else t)
        return torch.stack(result)

    elif attack == "JPEG":
        result = []
        for n in range(N):
            img = ((frames[n,:3]+1)/2*255).byte().permute(1,2,0).cpu().numpy()
            _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(param)])
            dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(dec).permute(2,0,1).float()/255*2-1
            result.append(torch.cat([t, frames[n,3:].cpu()], 0) if C==6 else t)
        return torch.stack(result)

    elif attack == "RO":
        result = []
        for n in range(N):
            img = TF.to_pil_image(torch.clamp((frames[n,:3]+1)/2, 0, 1))
            img = TF.rotate(img, float(param), expand=False)
            t = TF.to_tensor(img)*2-1
            result.append(torch.cat([t, frames[n,3:]], 0) if C==6 else t)
        return torch.stack(result)

    elif attack == "AF":
        result = []
        for n in range(N):
            img = TF.to_pil_image(torch.clamp((frames[n,:3]+1)/2, 0, 1))
            img = TF.affine(img, angle=10, translate=[10,10], scale=float(param), shear=0)
            t = TF.to_tensor(img)*2-1
            result.append(torch.cat([t, frames[n,3:]], 0) if C==6 else t)
        return torch.stack(result)

    return frames


class RobustnessDataset(Dataset):
    def __init__(self, base, attack, severity):
        self.base = base; self.attack = attack; self.severity = severity
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        s = self.base[idx]
        if self.severity == 0: return s
        return {**s, "frames": apply_attack(s["frames"].clone(), self.attack, self.severity)}


def resolve_model_path():
    for c in MODEL_CANDIDATES:
        if not os.path.exists(c): continue
        sd = torch.load(c, map_location="cpu")
        if all(torch.isfinite(v).all() for v in sd.values()): return c
    raise FileNotFoundError("No healthy checkpoint.")


# ── Charts ────────────────────────────────────────────────────────────────────

def save_heatmap(results, save_path, model_name="AFAGNetV3"):
    attacks = list(ATTACK_PARAMS.keys())
    sevs    = [0,1,2,3,4,5]
    data    = np.array([[results.get((a,s),{}).get("AUC",0)*100 for s in sevs] for a in attacks])
    fig, ax = plt.subplots(figsize=(10,5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="AUC (%)")
    ax.set_xticks(range(6)); ax.set_xticklabels(["Clean"]+[f"Sev {s}" for s in range(1,6)], fontsize=9)
    ax.set_yticks(range(len(attacks))); ax.set_yticklabels([ATTACK_NAMES[a] for a in attacks], fontsize=9)
    ax.set_title(f"{model_name} — Robustness AUC (%) Heatmap", fontsize=11, fontweight="bold")
    for i in range(len(attacks)):
        for j in range(6):
            v = data[i,j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white" if v<65 else "black")
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return save_path


def save_severity_curves(results, save_path, model_name="AFAGNetV3"):
    attacks = list(ATTACK_PARAMS.keys())
    colors  = plt.cm.tab10(np.linspace(0,1,len(attacks)))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for idx, atk in enumerate(attacks):
        ax   = axes[idx]
        aucs = [results.get((atk,s),{}).get("AUC",0)*100 for s in range(6)]
        ax.plot(range(6), aucs, marker="o", color=colors[idx], linewidth=2, markersize=5)
        ax.axhline(aucs[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylim(max(0, min(aucs)-5), 102)
        ax.set_xticks(range(6)); ax.set_xticklabels(["C","1","2","3","4","5"], fontsize=8)
        ax.set_title(ATTACK_NAMES[atk], fontsize=9, fontweight="bold")
        ax.set_ylabel("AUC (%)", fontsize=8); ax.grid(alpha=0.3)
        for s, v in enumerate(aucs):
            ax.text(s, v+0.5, f"{v:.1f}", ha="center", fontsize=6.5)
    fig.suptitle(f"{model_name} — AUC vs Attack Severity (C=Clean)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return save_path


def save_comparison_bar(mean_aucs, save_path, model_name="AFAGNetV3"):
    attacks   = list(ATTACK_PARAMS.keys())
    methods   = list(PAPER_RESULTS.keys()) + [model_name]
    all_data  = {**PAPER_RESULTS, model_name: mean_aucs}
    n_m, n_a  = len(methods), len(attacks)
    x, width  = np.arange(n_a), 0.75/n_m
    colors    = ["#94A3B8","#64748B","#475569","#334155","#1E293B","#2563EB"]
    fig, ax   = plt.subplots(figsize=(14,6))
    for i, mth in enumerate(methods):
        vals   = [all_data[mth].get(a,0) for a in attacks]
        offset = (i - n_m/2 + 0.5)*width
        bars   = ax.bar(x+offset, vals, width*0.9, label=mth,
                        color=colors[i%len(colors)], alpha=0.9, edgecolor="white", linewidth=0.5)
        if mth == model_name:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold", color="#2563EB")
    min_v = min(v for d in all_data.values() for v in d.values() if v>0)-5
    ax.set_ylim(max(0,min_v), 102)
    ax.set_xticks(x); ax.set_xticklabels([ATTACK_NAMES[a] for a in attacks], fontsize=9, rotation=10)
    ax.set_ylabel("Mean AUC (%) over severities 1-5", fontsize=10)
    ax.set_title("Robustness Comparison — 8 Attack Types (Zhou et al., ESWA 2025 Table 4 Style)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return save_path


def save_table_image(results, mean_aucs, save_path, model_name="AFAGNetV3"):
    attacks = list(ATTACK_PARAMS.keys())
    col_labels = ["Attack","Clean"]+[f"Sev{s}" for s in range(1,6)]+["Mean"]
    rows = []
    for atk in attacks:
        clean = results.get((atk,0),{}).get("AUC",0)*100
        sevs  = [results.get((atk,s),{}).get("AUC",0)*100 for s in range(1,6)]
        rows.append([ATTACK_NAMES[atk],f"{clean:.2f}"]+[f"{v:.2f}" for v in sevs]+[f"{mean_aucs[atk]:.2f}"])

    fig, ax = plt.subplots(figsize=(14,4))
    ax.axis("off")
    ax.set_title(f"{model_name} — Robustness Evaluation (Table 4 Format, ESWA 2025)",
                 fontsize=12, fontweight="bold", pad=12)

    col_w  = [0.18]+[0.09]*8
    x_pos  = [sum(col_w[:i]) for i in range(len(col_labels))]
    hy     = 0.92
    rh     = 0.78/(len(rows)+1)

    for j,col in enumerate(col_labels):
        ax.text(x_pos[j]+col_w[j]/2, hy, col, ha="center", va="top",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.plot([0,1],[hy-0.03]*2, color="#333", lw=1, transform=ax.transAxes, clip_on=False)

    for i,row in enumerate(rows):
        y  = hy-(i+1)*rh
        bg = "#F0F7FF" if i%2==0 else "white"
        rect = FancyBboxPatch((0,y-rh*0.85),1,rh*0.85, boxstyle="square,pad=0",
                               lw=0, facecolor=bg, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        for j,val in enumerate(row):
            col = "black"
            if j>0:
                try:
                    n = float(val)
                    col = "#15803D" if n>=95 else "#1D4ED8" if n>=85 else "#B45309" if n>=70 else "#DC2626"
                except: pass
            ax.text(x_pos[j]+col_w[j]/2, y-rh*0.4, val, ha="center", va="center",
                    fontsize=8.5, color=col, transform=ax.transAxes)

    ax.plot([0,1],[hy-(len(rows)+0.8)*rh]*2, color="#333", lw=1, transform=ax.transAxes, clip_on=False)
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return save_path


def main():
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(ROOT)
    _, val_indices = build_identity_disjoint_split(video_paths, val_ratio=0.2, seed=42)

    base_dataset = FFPPDatasetV2(
        video_paths=video_paths, labels=labels, mask_paths=mask_paths, domains=domains,
        cache_dir=CACHE_DIR, use_alignment=True, training_mode=False,
        label_smoothing=0.0, use_sbi=False,
    )
    val_subset = Subset(base_dataset, val_indices)

    model_path = resolve_model_path()
    model = AFAGNetV3().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    attacks    = list(ATTACK_PARAMS.keys())
    severities = [0,1,2,3,4,5]
    results    = {}
    total      = len(attacks)*len(severities)
    run        = 0

    print(f"Checkpoint : {model_path}")
    print(f"Eval size  : {len(val_subset)}")
    print(f"Total runs : {total} ({len(attacks)} attacks x {len(severities)} severities)\n")

    for atk in attacks:
        for sev in severities:
            run += 1
            lbl = "Clean" if sev==0 else f"Sev{sev}"
            print(f"  [{run:>3}/{total}] {ATTACK_NAMES[atk]:<22} {lbl} ...", end="", flush=True)
            ds = RobustnessDataset(val_subset, atk, sev)
            loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
            m = evaluate(model, loader, device=DEVICE, return_details=False, threshold=0.5)
            results[(atk,sev)] = m
            print(f"  AUC={m['AUC']*100:.2f}%  Acc={m['Accuracy']*100:.2f}%")

    # Mean AUC over severities 1-5
    mean_aucs = {atk: float(np.mean([results[(atk,s)]["AUC"]*100 for s in range(1,6)]))
                 for atk in attacks}

    # Console Table 4 style
    print(f"\n{'='*90}")
    print(f"  MEAN AUC (%) OVER SEVERITIES 1-5  —  Table 4 Format (ESWA 2025)")
    print(f"{'='*90}")
    print(f"  {'Method':<16} " + " ".join(f"{a:>7}" for a in attacks))
    print(f"  {'─'*16} " + " ".join("─"*7 for _ in attacks))
    for mth, vals in PAPER_RESULTS.items():
        print(f"  {mth:<16} " + " ".join(f"{vals.get(a,0):>7.2f}" for a in attacks))
    print(f"  {'─'*16} " + " ".join("─"*7 for _ in attacks))
    print(f"  {'AFAGNetV3(Ours)':<16} " + " ".join(f"{mean_aucs[a]:>7.2f}" for a in attacks))
    print(f"{'='*90}")

    print(f"\n  Gap vs MH-FFNet (SOTA):")
    for atk in attacks:
        diff = mean_aucs[atk] - PAPER_RESULTS["MH-FFNet"].get(atk,0)
        print(f"    {ATTACK_NAMES[atk]:<22}: {mean_aucs[atk]:.2f}%  (gap {diff:+.2f}%)")

    # Save charts
    save_dir = os.path.join("eval","robustness")
    os.makedirs(save_dir, exist_ok=True)
    save_heatmap(results,       os.path.join(save_dir,"robustness_heatmap.png"))
    save_severity_curves(results, os.path.join(save_dir,"robustness_severity_curves.png"))
    save_comparison_bar(mean_aucs, os.path.join(save_dir,"robustness_comparison_bar.png"))
    save_table_image(results, mean_aucs, os.path.join(save_dir,"robustness_table.png"))
    print(f"\n  Charts saved to {save_dir}/")
    print(f"    robustness_heatmap.png          — colour grid of AUC per severity")
    print(f"    robustness_severity_curves.png  — AUC vs severity curves (Fig.8 style)")
    print(f"    robustness_comparison_bar.png   — comparison with paper methods")
    print(f"    robustness_table.png            — full numeric table (Table 4 style)")


if __name__ == "__main__":
    main()
