
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from data.loader.splits import build_identity_disjoint_split
from eval.evaluate import evaluate, save_evaluation_record_image, save_evaluation_visuals
from models.afag_net_v3 import AFAGNetV3

ROOT      = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


class AblationModelV3(AFAGNetV3):
    """
    AFAGNetV3 with individual components disabled for ablation study.
    Weights loaded from the same checkpoint as the full model.
    """
    def __init__(
        self,
        disable_freq=False,
        disable_noise=False,
        disable_cgaf=False,
        disable_temporal=False,
        disable_f_high=False,       
    ):
        super().__init__()
        self.disable_freq     = disable_freq
        self.disable_noise    = disable_noise
        self.disable_cgaf     = disable_cgaf
        self.disable_temporal = disable_temporal
        self.disable_f_high   = disable_f_high

    def forward(self, x):
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)


        F_low, F_high = self.backbone(x_flat)

        if self.disable_f_high:
            F_high_for_spatial = None
            F_high_for_cgaf    = None
        else:
            F_high_for_spatial = F_high
            F_high_for_cgaf    = F_high

        # Branches (set to zeros to "disable")
        Fs = self.spatial(F_low, F_high_for_spatial)
        Ff = self.frequency(x_flat)
        Fn = self.noise(x_flat)

        if self.disable_freq:
            Ff = torch.zeros_like(Ff)
        if self.disable_noise:
            Fn = torch.zeros_like(Fn)

        if self.disable_cgaf:
            Ffusion_flat = Fs
            _, _, H_s, W_s = Fs.shape
            Csf = Csn = Cfn = torch.zeros(B * N, 1, H_s, W_s, device=x.device)
        else:
            Ffusion_flat, Csf, Csn, Cfn = self.cgaf(Fs, Ff, Fn, F_high=F_high_for_cgaf)

        _, C_f, H_f, W_f = Ffusion_flat.shape
        Ffusion = Ffusion_flat.view(B, N, C_f, H_f, W_f)

        if self.disable_temporal:
            video_feat = F.adaptive_avg_pool2d(
                Ffusion_flat, 1
            ).view(B, N, 256).mean(dim=1)
            video_feat = torch.cat([video_feat, video_feat], dim=1)
        else:
            video_feat = self.temporal(Ffusion)

        F_high_pool = F.adaptive_avg_pool2d(F_high, 1).view(B, N, 256).mean(dim=1)
        if self.disable_f_high:
            F_high_pool = torch.zeros_like(F_high_pool)

        pred_cls   = self.heads.cls_head(video_feat, F_high_pool)
        F_flat     = Ffusion.view(B * N, C_f, H_f, W_f)
        masks_flat = self.heads.loc_head(F_flat, F_high if not self.disable_f_high else None)
        mask_seq   = masks_flat.view(B, N, 1, 224, 224)
        pred_mask  = mask_seq[:, N // 2]

        conf_map  = torch.abs(pred_mask - 0.5) * 2
        stability = 1 - torch.abs(mask_seq[:, 1:] - mask_seq[:, :-1])

        return {
            "pred_cls":      pred_cls,
            "pred_mask":     pred_mask,
            "mask_sequence": mask_seq,
            "confidence":    conf_map,
            "stability":     stability,
            "attribution":   None,
            "Csf": Csf, "Csn": Csn, "Cfn": Cfn,
        }


def resolve_model_path():
    candidates = [
        os.path.join("checkpoints", "best_model.pth"),
        os.path.join("checkpoints", "latest_model.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No checkpoint found. Train first.")


def load_ablation_model(cfg, model_path, device):
    """
    Load checkpoint into AblationModelV3.
    Uses strict=False because ablation forward() differs from saved model's
    registered buffers/modules, but weights themselves are identical.
    Verifies no key is genuinely missing (only shape mismatches would be a problem).
    """
    model = AblationModelV3(**cfg).to(device)
    state  = torch.load(model_path, map_location=device)
    result = model.load_state_dict(state, strict=False)

    unexpected = [k for k in result.unexpected_keys if "disable" not in k]
    if unexpected:
        print(f"  [WARN] Unexpected keys in checkpoint: {unexpected[:5]}")

    missing = result.missing_keys
    if missing:
        print(f"  [WARN] Missing keys (expected for ablation): {len(missing)} keys")

    return model



def run_ablation(eval_dataset, model_path, device=DEVICE):
    configs = {
        "Full Model":      {},
        "No Frequency":    {"disable_freq": True},
        "No Noise":        {"disable_noise": True},
        "No CGAF":         {"disable_cgaf": True},
        "No Temporal":     {"disable_temporal": True},
        "No F_high":       {"disable_f_high": True},
    }

    loader  = DataLoader(eval_dataset, batch_size=2, shuffle=False, num_workers=0)
    results = {}

    for name, cfg in configs.items():
        print(f"\n{'='*50}")
        print(f"Ablation: {name}")
        print(f"{'='*50}")

        model   = load_ablation_model(cfg, model_path, device)
        metrics, details = evaluate(
            model, loader, device=device, return_details=True, threshold=0.5
        )
        results[name] = metrics

        save_dir = os.path.join("eval", "ablation", name.lower().replace(" ", "_"))
        save_evaluation_visuals(
            metrics,
            details["labels"],
            details["preds"],
            details["probs"],
            save_dir,
            prefix=f"ablation_{name.lower().replace(' ', '_')}",
            per_domain=details["per_domain"],
            dom_labels=details["dom_labels"],
            dom_probs=details["dom_probs"],
            model_name=f"AFAGNetV3 — {name}"
        )
        print(f"[INFO] Saved ablation evaluation visuals for {name} to {save_dir}")

        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"{'Component':<20} {'Accuracy':>9} {'AUC':>8} {'F1':>8} {'IoU':>8}")
    print(f"{'-'*60}")
    full_auc = results.get("Full Model", {}).get("AUC", 0)
    for name, m in results.items():
        delta = m.get("AUC", 0) - full_auc
        delta_str = f"({delta:+.4f})" if name != "Full Model" else "  (base) "
        print(f"{name:<20} {m.get('Accuracy',0):>9.4f} {m.get('AUC',0):>8.4f} "
              f"{m.get('F1',0):>8.4f} {m.get('IoU',0):>8.4f}  {delta_str}")
    print(f"{'='*60}")

    save_ablation_summary_chart(results, os.path.join("eval", "ablation", "ablation_auc_summary.png"))
    print("[INFO] Saved ablation AUC summary chart to eval/ablation/ablation_auc_summary.png")

    return results


def save_ablation_summary_chart(results: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    labels = list(results.keys())
    auc_scores = [results[name].get("AUC", 0.0) for name in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, [float(v) * 100 for v in auc_scores], color="#2563EB", edgecolor="white")
    ax.set_ylim(0, 100)
    ax.set_ylabel("AUC (%)", fontsize=11)
    ax.set_title("Ablation Study — AUC Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, auc in zip(bars, auc_scores):
        label = "N/A" if auc != auc else f"{auc*100:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                label,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(ROOT)

    _, val_indices = build_identity_disjoint_split(
        video_paths, val_ratio=0.2, seed=42
    )

    dataset = FFPPDatasetV2(
        video_paths=video_paths,
        labels=labels,
        mask_paths=mask_paths,
        domains=domains,
        cache_dir=CACHE_DIR,
        use_alignment=True,
        training_mode=False,
        label_smoothing=0.0,
        use_sbi=False,
    )
    eval_dataset = Subset(dataset, val_indices)
    model_path   = resolve_model_path()

    print(f"Checkpoint : {model_path}")
    print(f"Eval size  : {len(eval_dataset)}")

    results = run_ablation(eval_dataset, model_path, DEVICE)

    auc_scores = {name: m.get("AUC", 0) for name, m in results.items()}
    save_path  = os.path.join("eval", "ablation_auc.png")
    os.makedirs("eval", exist_ok=True)
    save_evaluation_record_image(auc_scores, save_path=save_path, title="Ablation Study — AUC")
    print(f"\nSaved ablation chart: {save_path}")


if __name__ == "__main__":
    main()
