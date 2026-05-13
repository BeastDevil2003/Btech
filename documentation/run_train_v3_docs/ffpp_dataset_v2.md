# ffpp_dataset_v2.py

## Purpose

`data/loader/ffpp_dataset_v2.py` implements the FF++ dataset class with alignment, caching, augmentation, and temporal sampling. It is the main data pipeline used by `run_train_v3.py`.

## Main class

### `FFPPDatasetV2(Dataset)`

This dataset class supports:

- Face detection and alignment.
- Cache building and loading.
- Training-time augmentation.
- Label smoothing.
- SBI augmentation.
- Temporal sampling strategies.

## Core behavior

### `__init__(...)`

- Stores video metadata: paths, labels, masks, domains.
- Configures cache directory and processing options.
- Sets up transform to resize frames to `224x224`.

### `__len__()`

- Returns the number of samples.

### `__getitem__(idx)`

- Loads a sample from cache if available.
- Otherwise builds the sample with `build_sample(idx)`.
- Applies augmentations only when `training_mode=True`.
- Applies label smoothing in training.

### `init_face_detector()`

- Initializes InsightFace face detector.
- Uses CPU or CUDA depending on environment.

### `cache_signature(idx)`

- Creates a hash key based on sample metadata and preprocessing options.
- Ensures cached files reflect the correct dataset configuration.

### `get_cache_path(idx)` and `save_cached_sample(cache_path, sample)`

- Manage cached `.pt` files safely.
- Save CPU tensors to avoid device-specific state.

## Additional utilities

### `_get_augmenters()`

- Lazily loads augmentation modules from `data.augmentation`.
- Supports `VideoAugmentation` and `SBIAugmentation` if available.

### Face alignment helpers

- `align_face_5pt()` aligns images by 5 facial landmarks.
- Uses `cv2.estimateAffinePartial2D` and falls back to center-cropping if alignment fails.

## Inputs

- `video_paths`, `labels`, `mask_paths`, `domains`: dataset metadata.
- `cache_dir`: directory for cached samples.
- `use_alignment`, `training_mode`, `label_smoothing`, `use_sbi`, `temporal_strategy`.

## Outputs

- Each item returns a dictionary containing:
  - `frames`: aligned video tensor
  - `mask_frames`: mask tensor
  - `label`: class label or smoothed label
  - `has_mask`: indicates if ground-truth mask is available
  - `domain`: sample domain

## Role in training

- Used by `run_train_v3.py` for cache building and dataset creation.
- Provides the data fed into PyTorch `DataLoader` objects.
- Supports augmentation only during training while preserving cached inputs.

## Notes

- Cache building is crucial to avoid extremely slow on-the-fly face detection.
- The cache includes alignment and temporal sampling results.
- `training_mode=False` is enforced for validation and cache building.
