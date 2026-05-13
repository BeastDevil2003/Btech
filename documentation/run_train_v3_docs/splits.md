# splits.py

## Purpose

`data/loader/splits.py` creates identity-aware training and validation splits. It ensures that no identity appears in both the train and val sets, avoiding identity leakage that can inflate validation metrics.

## Main functions

### `extract_identity_tokens(video_path)`

- Extracts identity tokens from the video filename.
- Uses filename segments separated by `_` and selects numeric tokens.
- If no numeric tokens exist, falls back to the full stem.
- Returns a tuple of identity tokens.

### `build_identity_disjoint_split(video_paths, val_ratio=0.2, seed=42)`

- Groups videos by identity tokens.
- Shuffles identity groups using a fixed random seed.
- Selects a validation subset of identities.
- Produces `train_indices` and `val_indices`.

## Inputs

- `video_paths`: list of video file paths produced by `build_ffpp_dataset()`.
- `val_ratio`: fraction of identities assigned to validation.
- `seed`: deterministic seed for reproducible splits.

## Outputs

- `train_indices`: sorted indices for training.
- `val_indices`: sorted indices for validation.

## Role in training

- Called by `run_train_v3.py` to create disjoint train/val datasets.
- Ensures evaluation measures generalization to unseen identities.
- `Subset` wrappers in `run_train_v3.py` use these indices to select samples.

## Notes

- Identity extraction is filename-driven, so dataset naming conventions matter.
- The split is deterministic given the seed.
- If no identity groups are found, the function raises an exception.
