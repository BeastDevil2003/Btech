import torch
import warnings

from data.loader.build_cache import build_cache
from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset import FFPPDataset
from data.loader.splits import build_identity_disjoint_split
from models.afag_net import AFAGNet
from torch.utils.data import Subset
from train.train_pipeline_final import train

warnings.filterwarnings(
    "ignore",
    message=".*HF Hub.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*rcond parameter will change.*",
    category=FutureWarning,
)

def main():
    root = "FaceForensics_Data"
    cache_dir = "dataset/cache/ffpp_processed"

    build_cache(root=root, cache_dir=cache_dir)

    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)

    dataset = FFPPDataset(
        video_paths,
        labels,
        mask_paths,
        domains,
        cache_dir=cache_dir,
    )
    print(f"Using cache directory: {cache_dir}")
    cache_stats = dataset.get_cache_stats()
    print(
        f"Training dataset cache status: "
        f"{cache_stats['cached']}/{cache_stats['total']} cached, "
        f"{cache_stats['missing']} missing"
    )

    train_indices, val_indices = build_identity_disjoint_split(video_paths, val_ratio=0.2, seed=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(
        f"Identity-aware split: train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples"
    )

    model = AFAGNet()

    train(model, train_dataset, val_dataset=val_dataset, epochs=10, batch_size=2, num_workers=0)


if __name__ == "__main__":
    main()
