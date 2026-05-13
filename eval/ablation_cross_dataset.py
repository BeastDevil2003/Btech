import os
import re
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from eval.ablation import load_ablation_model
from eval.evaluate import evaluate

MODEL_CANDIDATES = [
    os.path.join("checkpoints", "best_model.pth"),
    os.path.join("checkpoints", "latest_model.pth"),
    "model.pth",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOTS = {
    "FFPP": "FaceForensics_Data",
    "Celeb-DF": "path/to/celeb_df",
    "Celeb-DF-V2": "path/to/celeb_df_v2",
    "DFDC": "path/to/DFDC",
    "DFR": "path/to/DFR",
}


def build_dataset_from_root(name, root):
    if name == "FFPP":
        video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)
        return FFPPDatasetV2(
            video_paths=video_paths,
            labels=labels,
            mask_paths=mask_paths,
            domains=domains,
            cache_dir=None,
            use_alignment=True,
            training_mode=False,
            label_smoothing=0.0,
            use_sbi=False,
        )

    raise NotImplementedError(
        f"Dataset builder not implemented for {name}. "
        "Add a loader function if you want to evaluate this dataset."
    )


def resolve_model_path():
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No trained checkpoint found. Expected one of: "
        + ", ".join(MODEL_CANDIDATES)
    )


def select_dataset_names(available_names):
    print("[INFO] Available dataset roots:")
    for idx, name in enumerate(available_names, start=1):
        print(f"  {idx}. {name}")
    print("  0. All datasets")

    choice = input("Select dataset number(s) or name(s) separated by comma (default: all): ").strip()
    if not choice:
        return available_names

    tokens = [tok.strip() for tok in re.split(r"[\s,]+", choice) if tok.strip()]
    selected = []

    for token in tokens:
        if token == "0":
            return available_names
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(available_names):
                name = available_names[idx - 1]
                if name not in selected:
                    selected.append(name)
                continue
        if token in available_names and token not in selected:
            selected.append(token)
        else:
            print(f"[WARN] Ignored unknown dataset selection: {token}")

    if not selected:
        raise ValueError("No valid dataset selected. Please choose one of the listed datasets.")
    return selected


def save_cross_ablation_auc_chart(results, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    variant_names = list(next(iter(results.values())).keys())
    datasets = list(results.keys())

    x = range(len(variant_names))
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_idx, dataset_name in enumerate(datasets):
        aucs = [results[dataset_name][variant].get("AUC", 0.0) * 100 for variant in variant_names]
        ax.plot(
            x,
            aucs,
            marker="o",
            linewidth=2,
            label=dataset_name,
            alpha=0.85,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(variant_names, rotation=45, ha="right")
    ax.set_ylabel("AUC (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cross-Dataset Ablation Study — AUC (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_ablation_cross_dataset(selected_roots, model_path, device=DEVICE):
    configs = {
        "Full Model":      {},
        "No Frequency":    {"disable_freq": True},
        "No Noise":        {"disable_noise": True},
        "No CGAF":         {"disable_cgaf": True},
        "No Temporal":     {"disable_temporal": True},
        "No F_high":       {"disable_f_high": True},
    }

    results = {}

    for dataset_name, root in selected_roots.items():
        print(f"\n[INFO] Building dataset for {dataset_name} from {root}")
        dataset = build_dataset_from_root(dataset_name, root)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        results[dataset_name] = {}
        for variant_name, cfg in configs.items():
            print(f"\n[INFO] Evaluating {dataset_name} — {variant_name}")
            model = load_ablation_model(cfg, model_path, device)
            metrics = evaluate(model, loader, device=device, threshold=0.5)
            results[dataset_name][variant_name] = metrics
            print(f"  AUC: {metrics.get('AUC', 0.0):.4f}")
            if device == "cuda":
                torch.cuda.empty_cache()

    chart_path = os.path.join("eval", "ablation_cross_dataset_auc.png")
    save_cross_ablation_auc_chart(results, chart_path)
    print(f"\n[INFO] Saved cross-dataset ablation AUC chart to {chart_path}")
    return results


def main():
    available_roots = {
        name: root
        for name, root in DATASET_ROOTS.items()
        if os.path.exists(root)
    }

    if not available_roots:
        raise FileNotFoundError(
            "No dataset roots found. Update DATASET_ROOTS with actual paths before running."
        )

    selected_names = select_dataset_names(list(available_roots.keys()))
    selected_roots = {name: available_roots[name] for name in selected_names}
    model_path = resolve_model_path()
    run_ablation_cross_dataset(selected_roots, model_path, device=DEVICE)


if __name__ == "__main__":
    main()
