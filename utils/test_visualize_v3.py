import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset

from models.afag_net_v3 import AFAGNetV3
from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from data.loader.splits import build_identity_disjoint_split
from utils.visualize_v3 import visualize_sample

ROOT      = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_path():
    candidates = [
        os.path.join("checkpoints", "best_model.pth"),
        os.path.join("checkpoints", "ema_model.pth"),
    ]
    for c in candidates:
        if not os.path.exists(c):
            continue
        sd = torch.load(c, map_location="cpu")
        if all(torch.isfinite(v).all() for v in sd.values()):
            return c
    raise FileNotFoundError("No healthy checkpoint found. Train first.")


def main():
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(ROOT)

    real_idxs = [i for i, l in enumerate(labels) if l == 0][:2]
    fake_idxs = [i for i, l in enumerate(labels) if l == 1][:2]
    sample_indices = real_idxs + fake_idxs

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
    subset = Subset(dataset, sample_indices)
    loader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=0)

    model_path = resolve_model_path()
    print(f"Loading checkpoint: {model_path}")

    model = AFAGNetV3().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    batch = next(iter(loader))
    frames = batch["frames"].to(DEVICE)

    with torch.no_grad():
        output = model(frames)

    output_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in output.items()
    }
    batch_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    visualize_sample(batch_cpu, output_cpu, save_path="outputs/test_viz")
    print("\nDone. Check outputs/test_viz/")


if __name__ == "__main__":
    main()