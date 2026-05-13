# train_pipeline_v3.py

## Purpose

`train/train_pipeline_v3.py` contains the full training and validation engine for AFAGNet v3. It implements dataset loaders, sampling, loss functions, optimization, checkpointing, metrics, and graph generation.

## Main responsibilities

- Creates `DataLoader` objects for train and validation datasets.
- Builds a balanced sampler to counter label imbalance.
- Initializes optimizer, scheduler, and AMP scaler.
- Runs training and validation loops.
- Saves best/latest model checkpoints and EMA weights.
- Logs epoch results and generates graphs.

## Key components

### `plot_training_history(history, save_dir, batch_size)`

- Generates plots for loss, AUC, accuracy, learning rate, and overfitting.
- Saves PNG files to the log graph directory.

### `FocalLoss(nn.Module)`

- Binary focal loss with alpha and gamma settings.
- Used to prioritize minority class gradients.

### `branch_consistency_loss(Fs, Ff, Fn)`

- Encourages feature agreement across spatial, frequency, and noise branches.
- Uses cosine similarity over pooled branch features.

### `localization_loss(pred_mask, gt_mask, has_mask, pseudo_mask=None)`

- Computes mask supervision loss for real masks.
- Adds pseudo-mask loss for unlabeled samples.

### `temporal_consistency_loss(mask_seq)`

- Penalizes large changes between consecutive mask frames.

### `mixup_batch(frames, labels, alpha)` and `mixup_loss(...)`

- Implements video-level MixUp augmentation.
- Mixes both the frame tensors and labels.

### `ModelEMA`

- Maintains an exponential moving average of model weights.
- Used for validation and stable checkpointing.

### `WarmupScheduler`

- Warms up learning rate over a few epochs.
- Then hands off to cosine annealing.

### `make_balanced_sampler(dataset)`

- Builds a sample weight list from dataset labels.
- Supports `FFPPDatasetV2` and `Subset`.

### `train_one_epoch(...)`

- Runs a single training epoch.
- Applies MixUp after warmup and starting from epoch 3.
- Computes classification, localization, consistency, and temporal losses.
- Uses gradient scaling and clipping.
- Updates EMA weights if enabled.

### `validate(...)`

- Runs validation in evaluation mode.
- Computes AUC and accuracy.
- Tracks per-domain metrics.

### `train(...)`

- Full training loop.
- Saves checkpoints and logs.
- Implements early stopping on validation AUC.
- Generates final training graphs.

## Inputs

- `model`: AFAGNetV3 instance.
- `train_dataset`, `val_dataset`: `FFPPDatasetV2` subsets.
- `epochs`, `batch_size`, `lr`, `weight_decay`, `num_workers`, etc.
- `resume_epoch`: starting epoch when resuming training.

## Outputs

- Saves model weights to `checkpoints/best_model.pth`, `latest_model.pth`, `ema_model.pth`.
- Writes epoch metrics to `experiments/logs/epoch_results_v3.txt`.
- Saves training graphs under `experiments/logs/graphs/`.
- Returns `history` dictionary with metrics for every epoch.

## Role in training

- Called by `run_train_v3.py` to perform the actual optimization.
- Contains the logic for the end-to-end training lifecycle.

## Notes

- Training uses mixed precision (`use_amp=True`) on CUDA.
- The pipeline is robust to NaN batches and avoids saving corrupt checkpoint states.
- EMA is only saved when the epoch is clean.
