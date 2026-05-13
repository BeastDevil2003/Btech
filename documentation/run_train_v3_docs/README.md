# run_train_v3.py Documentation

## Overview

`train/run_train_v3.py` is the training entrypoint for the AFAGNet v3 deepfake detection model. It performs end-to-end dataset preparation, cache building, model loading/resuming, and training. The script runs over the FaceForensics++ dataset structure under `FaceForensics_Data` and saves model checkpoints, logs, and graphs.

## What this file does

1. Detects GPU availability and selects a batch size using `detect_batch_size()`.
   - Prints detected GPU specifications and available `HPARAM_TABLE` profiles to help the user choose.
   - Supports command-line overrides such as `--batch-size`, `--lr`, `--weight-decay`, `--mixup-alpha`, `--warmup-epochs`, `--patience`, `--num-workers`, and `--focal-alpha`.
   - also option to make own customise profile
2. Loads the FaceForensics++ dataset metadata using `build_ffpp_dataset()`.
3. Builds a preprocessed cache of aligned face frames using `FFPPDatasetV2` if missing.
4. Verifies at least 80% of dataset samples are cached before training.
5. Creates an identity-disjoint train/validation split using `build_identity_disjoint_split()`.
6. Builds train and validation PyTorch datasets with appropriate augmentation and smoothing flags.
7. Creates the `AFAGNetV3` model and optionally resumes from an existing checkpoint.
8. Calls `train()` from `train/train_pipeline_v3.py` to optimize the model.
9. Prints the final best validation AUC after training.

## Inputs

- `FaceForensics_Data`: root dataset folder.
  - `original_sequences/youtube` and `original_sequences/actors` for real videos.
  - `manipulated_sequences/<fake_type>/c23` for fake videos.
  - `manipulated_sequences/<fake_type>/masks/videos` for mask paths when available.
- `checkpoints/best_model.pth`: optional checkpoint used to resume training.
- `experiments/logs/epoch_results_v3.txt`: optional log file used to infer the last completed epoch.

## Outputs

- `dataset/cache/ffpp_processed_v2`: cache files containing preprocessed samples.
- `checkpoints/best_model.pth`: saved best model checkpoint by validation AUC.
- `checkpoints/latest_model.pth`: latest checkpoint after every epoch.
- `checkpoints/ema_model.pth`: EMA-weighted model weights if EMA is enabled.
- `experiments/logs/epoch_results_v3.txt`: training log file with epoch metrics.
- `experiments/logs/graphs/training_history.png`: training history graph.
- `experiments/logs/graphs/overfit_monitor.png`: overfitting monitor graph.

## Training flow

### 1. GPU + hyperparameter selection

- If CUDA is available, the batch size is selected by VRAM:
  - `>= 14GB` → batch size 8
  - `>= 8GB` → batch size 4
  - otherwise → batch size 2
- Hyperparameters are chosen from `HPARAM_TABLE`.

### 2. Dataset building

- `build_ffpp_dataset(root)` scans dataset directories and returns:
  - `video_paths`: list of `.mp4` video file paths.
  - `labels`: `0` for real, `1` for fake.
  - `mask_paths`: path to mask video if available, else `None`.
  - `domains`: domain labels such as `youtube`, `actors`, or fake type.

- `FFPPDatasetV2(...)` is used to build a cache and later to load samples.
  - Face detection and alignment are applied.
  - Frame selection, normalization, and optional augmentations are applied.

- Cache build step uses `cache_builder.build_cache(num_workers=0)` on Windows. If number of worker want to increase then give value at line no 100
- The script aborts with `sys.exit(1)` if less than 80% of data is cached.

### 3. Train/Validation split

- `build_identity_disjoint_split(video_paths, val_ratio=0.2, seed=42)` groups videos by identity tokens extracted from filenames.
- Creates a split that avoids identity overlap between train and validation.

### 4. Dataset objects

- `train_dataset`: `FFPPDatasetV2` with `training_mode=True`, `label_smoothing=0.05`, `use_sbi=True`, augmentation enabled.
- `val_dataset`: `FFPPDatasetV2` with `training_mode=False`, `label_smoothing=0.0`, `use_sbi=False`, no augmentation.

### 5. Model initialization and resume logic

- Initializes `AFAGNetV3(use_gradient_checkpointing=False)`.
- If `checkpoints/best_model.pth` exists, loads weights and attempts to infer completed epochs from the log file.
- If training is already finished, stops early.

### 6. Training call

- Calls `train(...)` with:
  - model, datasets, `TOTAL_EPOCHS = 50`. to increase epoch set value at 156 line
  - hyperparameters from `HPARAM_TABLE`
  - `use_amp=True`, `use_mixup=True`, `use_ema=True`
  - save and log directories

## Called files and roles

This file called this files

- `data/loader/build_dataset.py`
  - collects FF++ videos and assigns labels/domains.
- `data/loader/ffpp_dataset_v2.py`
  - implements `FFPPDatasetV2` with cache, alignment, augmentation, and frame sampling.
- `data/loader/splits.py`
  - creates identity-disjoint train/validation splits.
- `models/afag_net_v3.py`
  - defines the AFAGNet v3 model architecture and output heads.
- `train/train_pipeline_v3.py`
  - implements the optimizer, scheduler, training loop, validation loop, loss functions, metrics, and checkpointing.
- `train/psuedo_mask.py`
  - generates pseudo masks used during training when ground-truth masks are unavailable.

## Notes

- `run_train_v3.py` is designed for Windows compatibility during cache building (`num_workers=0`).
- Augmentation code is optional and loaded only if `data.augmentation` imports successfully.
- The training pipeline uses Focal Loss, MixUp, EMA, and a cosine annealing learning rate schedule with warmup.
- The dataset cache is keyed by `cache_signature()` and includes all relevant preprocessing options.

## Where to read next

For deeper detail, see `called_files_details.md` in this folder.

### Per-file summaries

- [`build_dataset.md`](build_dataset.md)
- [`ffpp_dataset_v2.md`](ffpp_dataset_v2.md)
- [`splits.md`](splits.md)
- [`afag_net_v3.md`](afag_net_v3.md)
- [`train_pipeline_v3.md`](train_pipeline_v3.md)
- [`psuedo_mask.md`](psuedo_mask.md)
