import os
import re
import torch

from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDataset
from eval.evaluate import evaluate, save_evaluation_visuals
from eval.cross_dataset import run_cross_dataset
from models.afag_net_v3 import AFAGNetV3

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


def build_dataset_from_root(root):
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)
    return FFPPDataset(video_paths, labels, mask_paths, domains)


def select_dataset_names(available_names):
    print("[INFO] Available datasets for validation:")
    for idx, name in enumerate(available_names, start=1):
        print(f"  {idx}. {name}")
    print("  0. All datasets")

    choice = input("Enter dataset number(s) or name(s) separated by comma (default: all): ").strip()
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


def resolve_model_path():
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No trained checkpoint was found. Expected one of: "
        + ", ".join(MODEL_CANDIDATES)
    )


def main():
    available_roots = {
        name: root
        for name, root in DATASET_ROOTS.items()
        if os.path.exists(root)
    }

    if not available_roots:
        raise FileNotFoundError("No evaluation dataset roots were found. Update DATASET_ROOTS first.")

    selected_names = select_dataset_names(list(available_roots.keys()))
    datasets = {
        name: build_dataset_from_root(available_roots[name])
        for name in selected_names
    }

    model_path = resolve_model_path()
    results = run_cross_dataset(model_path, datasets, device=DEVICE)

    model = AFAGNetV3().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    for name, dataset in datasets.items():
        loader = torch.utils.data.DataLoader(dataset, batch_size=2) 
        metrics, details = evaluate(model, loader, device=DEVICE, return_details=True)
        save_dir = os.path.join("eval", "cross_dataset")
        prefix = name.lower().replace("+", "plus").replace(" ", "_")
        paths = save_evaluation_visuals(metrics, details["labels"], details["preds"], details["probs"], save_dir, prefix)
        print(f"[INFO] Saved visual records for {name}:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

    return results


if __name__ == "__main__":
    main()
