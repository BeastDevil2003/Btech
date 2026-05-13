import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from models.afag_net_v3 import AFAGNetV3
from eval.evaluate import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_CHECKPOINT = None
MODEL_CANDIDATES = [
    os.path.join("checkpoints", "best_model.pth"),
    os.path.join("checkpoints", "ema_model.pth"),
    os.path.join("checkpoints", "latest_model.pth"),
]


def is_weights_healthy(sd):
    for name, t in sd.items():
        if not torch.isfinite(t).all():
            return False, name
    return True, None


def resolve_model_path(checkpoint=None):
    if checkpoint is not None:
        if os.path.exists(checkpoint):
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if FORCE_CHECKPOINT:
        if os.path.exists(FORCE_CHECKPOINT):
            return FORCE_CHECKPOINT
        raise FileNotFoundError(f"FORCE_CHECKPOINT not found: {FORCE_CHECKPOINT}")
    for c in MODEL_CANDIDATES:
        if not os.path.exists(c):
            continue
        try:
            sd = torch.load(c, map_location="cpu")
            ok, bad = is_weights_healthy(sd)
            if ok:
                return c
            print(f"  WARNING: {c} has NaN/Inf in '{bad}' — skipping")
        except Exception as e:
            print(f"  WARNING: Could not load {c}: {e}")
    raise FileNotFoundError("No healthy checkpoint found.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AFAGNetV3 direct video inference / evaluation."
    )
    parser.add_argument(
        "--video", type=str, required=False,
        help="Direct path to a single video or image file. If omitted, the script will ask for it interactively."
    )
    parser.add_argument(
        "--label", type=int, choices=[0, 1], default=None,
        help="Optional ground truth label for the video/image (0=real, 1=fake)."
    )
    parser.add_argument(
        "--domain", type=str, default="youtube",
        help="Domain name for the input video."
    )
    parser.add_argument(
        "--mask", type=str, default=None,
        help="Optional mask video path for fake videos."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional checkpoint path to load instead of the default candidates."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for fake/real classification."
    )
    return parser.parse_args()


def build_single_video_dataset(video_path, label, domain, mask_path):
    if label is None:
        label = 0
    return [video_path], [label], [mask_path], [domain]


def extract_state_dict(loaded):
    if isinstance(loaded, dict):
        if "state_dict" in loaded:
            return loaded["state_dict"]
        if "model_state_dict" in loaded:
            return loaded["model_state_dict"]
    return loaded


def load_state_dict_safe(model, state_dict):
    model_dict = model.state_dict()
    compatible = {}
    skipped = []

    for name, param in state_dict.items():
        if name not in model_dict:
            skipped.append(name)
            continue
        if model_dict[name].shape != param.shape:
            skipped.append(name)
            continue
        compatible[name] = param

    if not compatible:
        raise RuntimeError(
            "No compatible weights were found in the checkpoint for the current model architecture. "
            "Please use a checkpoint that matches the current AFAGNetV3 version."
        )

    model.load_state_dict(compatible, strict=False)
    missing = [name for name in model_dict if name not in compatible]
    return compatible, skipped, missing


def load_model(checkpoint=None):
    model_path = resolve_model_path(checkpoint)
    model = AFAGNetV3().to(DEVICE)
    loaded = torch.load(model_path, map_location=DEVICE)
    state_dict = extract_state_dict(loaded)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        print("WARNING: strict state_dict load failed. Trying to load compatible parameter subsets...")
        if not isinstance(state_dict, dict):
            raise
        compatible, skipped, missing = load_state_dict_safe(model, state_dict)
        print(f"  Loaded {len(compatible)}/{len(model.state_dict())} compatible parameters.")
        print(f"  Skipped {len(skipped)} checkpoint keys due to mismatch or unexpected names.")
        print(f"  {len(missing)} model parameters were not initialized from checkpoint.")
        if len(compatible) == 0:
            raise RuntimeError(
                "Failed to load any compatible weights from checkpoint. "
                "Please use a matching checkpoint or update the model architecture."
            )

    model.eval()
    return model_path, model


def print_metrics(metrics, details):
    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {metrics.get('Accuracy', 0) * 100:.2f}%")
    print(f"Precision: {metrics.get('Precision', 0) * 100:.2f}%")
    print(f"Recall   : {metrics.get('Recall', 0) * 100:.2f}%")
    print(f"F1       : {metrics.get('F1', 0) * 100:.2f}%")
    print(f"AUC      : {metrics.get('AUC', 0) * 100:.2f}%")
    print(f"IoU      : {metrics.get('IoU', 0) * 100:.2f}%")
    if details is not None:
        print(f"Probability: {details['probs'][0] * 100:.2f}%")
        print(f"Prediction : {'FAKE' if details['preds'][0] == 1.0 else 'REAL'}")
        if details.get('labels') is not None:
            print(f"Label      : {int(details['labels'][0])}")
            print(f"Result     : {'correct' if details['labels'][0] == details['preds'][0] else 'incorrect'}")


def main():
    args = parse_args()

    if not args.video:
        args.video = input("Enter direct video path: ").strip()

    if not args.video:
        raise ValueError("Video path is required.")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.mask is not None and not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask video not found: {args.mask}")

    model_path, model = load_model(args.checkpoint)
    print(f"\nCheckpoint : {model_path}")
    print(f"Device     : {DEVICE}")
    print(f"Video      : {args.video}")

    video_paths, labels, mask_paths, domains = build_single_video_dataset(
        args.video, args.label, args.domain, args.mask
    )

    dataset = FFPPDatasetV2(
        video_paths=video_paths,
        labels=labels,
        mask_paths=mask_paths,
        domains=domains,
        cache_dir=None,
        use_alignment=True,
        training_mode=False,
        label_smoothing=0.0,
        use_sbi=False,
        temporal_strategy="blend",
    )

    if args.label is None:
        sample = dataset[0]
        frames = sample["frames"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(frames)
            prob = torch.sigmoid(out["pred_cls"].squeeze(1)).item()
        print(f"Predicted  : {'FAKE' if prob >= args.threshold else 'REAL'}")
        print(f"Probability: {prob * 100:.2f}%")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=False)

    metrics, details = evaluate(
        model, loader, device=DEVICE,
        return_details=True, threshold=args.threshold,
    )

    print_metrics(metrics, details)


if __name__ == "__main__":
    main()
