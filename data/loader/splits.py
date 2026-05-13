from pathlib import Path

import torch


def extract_identity_tokens(video_path):
    stem = Path(video_path).stem
    prefix = stem.split("__", 1)[0]
    numeric_tokens = [token for token in prefix.split("_") if token.isdigit()]
    if numeric_tokens:
        return tuple(sorted(set(numeric_tokens)))
    return (stem,)


def build_identity_disjoint_split(video_paths, val_ratio=0.2, seed=42):
    identity_to_indices = {}

    for idx, video_path in enumerate(video_paths):
        for token in extract_identity_tokens(video_path):
            identity_to_indices.setdefault(token, []).append(idx)

    identity_keys = sorted(identity_to_indices)
    if not identity_keys:
        raise ValueError("No identity groups were found for dataset splitting.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(identity_keys), generator=generator).tolist()
    target_val_identities = max(1, int(round(len(identity_keys) * val_ratio)))

    val_indices = set()
    for perm_idx in perm[:target_val_identities]:
        val_indices.update(identity_to_indices[identity_keys[perm_idx]])

    all_indices = set(range(len(video_paths)))
    train_indices = sorted(all_indices - val_indices)
    val_indices = sorted(val_indices)
    return train_indices, val_indices
