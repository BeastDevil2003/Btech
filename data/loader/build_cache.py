import os
import argparse
from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset import FFPPDataset

def build_cache(root="FaceForensics_Data", cache_dir="dataset/cache/ffpp_processed", num_workers=None):
    # 1. Gather paths
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)
    
    # 2. Correctly initialize the dataset object
    dataset = FFPPDataset(
        video_paths=video_paths, 
        labels=labels, 
        mask_paths=mask_paths, 
        domains=domains, 
        cache_dir=cache_dir
    )
    
    # 3. Set the worker count to protect your RTX 3050 VRAM
    worker_count = num_workers if num_workers is not None else 2

    print(f"Starting build with {worker_count} workers to protect VRAM...")
    
    # 4. Call the method on the 'dataset' object we just created
    return dataset.build_cache(num_workers=worker_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    build_cache(num_workers=args.workers)
















"""
import os
import warnings

from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset import FFPPDataset

warnings.filterwarnings(
    "ignore",
    message=".*Error fetching version info.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*rcond parameter will change.*",
    category=FutureWarning,
)


def build_cache(
    root="FaceForensics_Data",
    cache_dir="dataset/cache/ffpp_processed",
    num_workers=None,
):
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)
    dataset = FFPPDataset(
        video_paths,
        labels,
        mask_paths,
        domains,
        cache_dir=cache_dir,
    )
    worker_count = (
        min(max(os.cpu_count() or 1, 1), 3)
        if num_workers is None else max(int(num_workers), 1)
    )

    stats = dataset.get_cache_stats()
    providers, _ = dataset.get_insightface_runtime()
    print(f"Building cache into: {cache_dir}")
    print(
        f"Cache status before build: "
        f"{stats['cached']}/{stats['total']} cached, "
        f"{stats['missing']} missing"
    )
    print(f"Face detector providers: {providers}")
    print(f"Cache workers: {worker_count}")

    build_stats = dataset.build_cache(num_workers=worker_count)
    print(
        f"Cache build summary: "
        f"built={build_stats['built']}, "
        f"reused={build_stats['skipped']}, "
        f"cached_now={build_stats['cached_after']}/{build_stats['total']}"
    )
    return build_stats


def main():
    build_cache()
    print("Cache build complete.")


if __name__ == "__main__":
    main()
"""