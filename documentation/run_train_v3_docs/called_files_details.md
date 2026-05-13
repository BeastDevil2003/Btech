# Called Files Detailed Summary

This file explains the files imported and invoked by `train/run_train_v3.py` and their responsibilities.

## 1. `data/loader/build_dataset.py`

### Function: `build_ffpp_dataset(root_dir, compression="c23")`

- Scans the FaceForensics++ dataset folder structure.
- Builds lists of video file paths, binary labels, mask paths, and domain names.
- Real videos:
  - `original_sequences/youtube`
  - `original_sequences/actors`
- Fake videos:
  - `manipulated_sequences/<fake_type>/c23`
  - `manipulated_sequences/<fake_type>/masks/videos`
- Returns:
  - `video_paths`: list of video file paths.
  - `labels`: `0` for real, `1` for fake.
  - `mask_paths`: path to mask `.mp4` for fake samples when available, otherwise `None`.
  - `domains`: string domain labels for each video.

## 2. `data/loader/ffpp_dataset_v2.py`

### Class: `FFPPDatasetV2(Dataset)`

This dataset class is the core of the data pipeline.

Responsibilities:
- Build and load cached preprocessed samples.
- Detect and align faces using InsightFace and MTCNN.
- Sample fixed-length video clips.
- Resize frames to 224×224 and convert to tensors.
- Apply training augmentations when `training_mode=True`.
- Support label smoothing and SBI augmentation.

### Main methods

- `__len__()`
  - Returns number of videos.
- `__getitem__(idx)`
  - Loads sample from cache or builds it.
  - Applies augmentation only during training.
  - Applies label smoothing in training mode.
- `init_face_detector()`
  - Prepares the InsightFace detector for face cropping.
- `build_cache(num_workers=0)`
  - Builds missing preprocessed cache files.
- `cache_signature(idx)`
  - Creates a deterministic cache key from sample metadata and processing options.
- `get_cache_path(idx)` and `save_cached_sample(cache_path, sample)`
  - Manage cached `.pt` files.

### Preprocessing details

- Face alignment uses 5-point landmark transformation.
- Output frames are normalized and resized.
- Temporal sampling strategy is `blend` by default.
- Mask frames and label tensors are stored in the cached sample.

## 3. `data/loader/splits.py`

### Function: `build_identity_disjoint_split(video_paths, val_ratio=0.2, seed=42)`

- Extracts identity tokens from each video filename.
- Groups videos by identity.
- Randomly selects validation identities.
- Returns train and validation index lists.
- Ensures the same identity is not present in both sets.

## 4. `models/afag_net_v3.py`

### Class: `AFAGNetV3(nn.Module)`

Defines the full AFAGNet v3 model used for training.

### Main submodules

- `AFAGBackbone()`
  - Produces low-level `F_low` and high-level `F_high` features.
- `SpatialBranch(channels=256, high_channels=256)`
  - Extracts spatial features from backbone outputs.
- `FrequencyBranch()`
  - Extracts frequency-domain features.
- `NoiseBranch()`
  - Extracts noise residual features.
- `CGAFv2(channels=256, use_f_high=True)`
  - Fuses branch features with gated adaptation.
- `TemporalModel()`
  - Aggregates temporal video features.
- `OutputHeads()`
  - `ClassificationHead` returns a fake/real logit.
  - `LocalizationHead` returns a manipulation mask for the center frame.

### Forward pass

- Input shape: `(B, N, 6, 224, 224)`
- Backbone outputs:
  - `F_low` → texture features
  - `F_high` → semantic context features
- Three branches compute:
  - `Fs`: spatial features
  - `Ff`: frequency features
  - `Fn`: noise features
- `CGAFv2` fuses them into `Ffusion`.
- `TemporalModel` aggregates video-level information.
- `OutputHeads` produce classification logits and localization masks.

## 5. `train/train_pipeline_v3.py`

### Function: `train(...)`

This is the main training engine called by `run_train_v3.py`.

Responsibilities:
- Sets dataset augmentation mode.
- Creates balanced sampler and data loaders.
- Moves model to device.
- Builds optimizer and learning rate scheduler.
- Creates mixed-precision scaler.
- Creates EMA model if enabled.
- Runs training and validation for each epoch.
- Saves checkpoints and logs.
- Generates training graphs.

### Optimization and checkpointing

- Uses `AdamW` optimizer.
- Separate learning rates for backbone vs rest of the model.
- `WarmupScheduler` for warmup followed by cosine annealing.
- `ModelEMA` maintains exponential moving averages of weights.
- Saves best model by validation AUC.
- Supports training resume with `resume_epoch`.

### Loss and metrics

- `FocalLoss(alpha=focal_alpha, gamma=focal_gamma)` for classification.
- `localization_loss()` for mask supervision and pseudo-mask guidance.
- `branch_consistency_loss()` encourages agreement between feature branches.
- `temporal_consistency_loss()` smooths predicted masks over time.
- `batch_per_class_accuracy()` reports balanced accuracy.
- Validation computes domain-specific metrics.

### Training loop behavior

- Uses `train_one_epoch()` per epoch.
- Optionally uses MixUp after warmup and after epoch 3.
- Validates using EMA weights if enabled.
- Skips saving best model if the epoch had NaN batches.
- Writes epoch metrics to `epoch_results_v3.txt`.
- Generates graphs in `experiments/logs/graphs`.

## 6. `train/psuedo_mask.py`

### Function: `generate_pseudo_mask(...)`

- Produces pseudo ground-truth masks for samples without real masks.
- Uses frequency and noise branch activations.
- Optionally uses GradCAM for a more informed mask.
- Returns normalized pseudo masks for the central frame.

### Usage in `train_pipeline_v3.py`

- Called only if the module is importable and if `Fs`, `Ff`, `Fn` are available.
- Used in `train_one_epoch()` to improve localization loss on samples without true masks.

## Summary

`train/run_train_v3.py` is a complete training script that coordinates:

- dataset discovery and label creation,
- face alignment and cache construction,
- identity-aware dataset splitting,
- model initialization and resume logic,
- training with advanced loss terms, augmentation, EMA, and checkpointing.

The critical helper files are documented above so you can follow the end-to-end pipeline from input video datasets to saved model weights and logs.
