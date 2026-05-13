# build_dataset.py

## Purpose

`data/loader/build_dataset.py` builds the video metadata used by `run_train_v3.py`. It scans the FaceForensics++ dataset structure and returns file paths, labels, mask references, and domain identifiers.

## Main function

### `build_ffpp_dataset(root_dir, compression="c23")`
-  <b>Note</b>: for use c40 dataset give value at line no 4 in build_dataset.py  `data/loader/build_dataset.py`
- <b>Note</b>: for use raw dataset give value at line no 4 in build_dataset.py  `data/loader/build_dataset.py`
- Scans the dataset directory tree under `root_dir`.
- Collects:
  - `video_paths`: all `.mp4` video file paths.
  - `labels`: `0` for real, `1` for fake.
  - `mask_paths`: per-video mask file paths when available.
  - `domains`: a domain string for each sample.

## Input data layout

The function assumes the FF++ folder structure:

- `original_sequences/youtube`
- `original_sequences/actors`
- `manipulated_sequences/<fake_type>/c23`
- `manipulated_sequences/<fake_type>/masks/videos`

`fake_type` comes from the subdirectories in `manipulated_sequences`.

## Output

Returns a tuple:

- `video_paths`: list[str]
- `labels`: list[int]
- `mask_paths`: list[Optional[str]]
- `domains`: list[str]

## Role in training

- `run_train_v3.py` calls this function to enumerate the full dataset.
- The returned lists are passed into `FFPPDatasetV2`.
- The `labels` and `mask_paths` arrays are critical for balanced sampling, mask supervision, and domain tracking.

## Behavior details

- Real videos are labeled as `0` and use `None` for `mask_paths`.
- Fake videos are labeled as `1`.
- For fake videos where masks are missing or not present (e.g. `FaceShifter`), `mask_paths` stores `None`.
- Domain names preserve the fake technique folder name, enabling per-domain validation statistics.
