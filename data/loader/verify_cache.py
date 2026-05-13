import torch

from data.loader.build_dataset import build_ffpp_dataset
from data.loader.ffpp_dataset_2 import FFPPDataset


def compare_tensors(name, a, b, atol=1e-6):
    same_shape = a.shape == b.shape
    max_abs_diff = float((a - b).abs().max().item()) if same_shape else None
    is_close = same_shape and torch.allclose(a, b, atol=atol, rtol=0.0)
    print(
        f"{name}: "
        f"same_shape={same_shape}, "
        f"allclose={is_close}, "
        f"max_abs_diff={max_abs_diff}"
    )
    return is_close


def main():
    root = "FaceForensics_Data"
    cache_dir = "dataset/cache/ffpp_processed"
    sample_idx = 0

    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)

    live_dataset = FFPPDataset(
        video_paths,
        labels,
        mask_paths,
        domains,
        cache_dir=None,
    )
    cached_dataset = FFPPDataset(
        video_paths,
        labels,
        mask_paths,
        domains,
        cache_dir=cache_dir,
    )

    cache_path = cached_dataset.get_cache_path(sample_idx)
    if not cache_path.exists():
        print(f"Cache missing for sample {sample_idx}. Building just this sample first.")
        sample = cached_dataset.build_sample(sample_idx)
        cached_dataset.save_cached_sample(cache_path, sample)

    print(f"Comparing sample index: {sample_idx}")
    print(f"Video path: {video_paths[sample_idx]}")

    live_sample = live_dataset.build_sample(sample_idx)
    cached_sample = cached_dataset[sample_idx]

    frames_ok = compare_tensors("frames", live_sample["frames"], cached_sample["frames"])
    masks_ok = compare_tensors("mask_frames", live_sample["mask_frames"], cached_sample["mask_frames"])
    label_ok = compare_tensors(
        "label",
        live_sample["label"].view(1),
        cached_sample["label"].view(1),
    )
    has_mask_ok = bool(live_sample["has_mask"]) == bool(cached_sample["has_mask"])
    domain_ok = live_sample["domain"] == cached_sample["domain"]

    print(f"has_mask match: {has_mask_ok}")
    print(f"domain match:   {domain_ok}")

    if frames_ok and masks_ok and label_ok and has_mask_ok and domain_ok:
        print("Cache verification PASSED.")
    else:
        print("Cache verification FAILED.")


if __name__ == "__main__":
    main()
