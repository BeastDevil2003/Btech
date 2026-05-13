import os, math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.loader.build_dataset   import build_ffpp_dataset
from data.loader.ffpp_dataset_v2 import FFPPDatasetV2
from data.loader.splits          import build_identity_disjoint_split
from models.afag_net_v3          import AFAGNetV3
from eval.evaluate               import evaluate, save_evaluation_visuals

ROOT      = "FaceForensics_Data"
CACHE_DIR = "dataset/cache/ffpp_processed_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

FORCE_CHECKPOINT = None

MODEL_CANDIDATES = [
    os.path.join("checkpoints", "best_model.pth"),
    os.path.join("checkpoints", "ema_model.pth"),
    os.path.join("checkpoints", "latest_model.pth"),
]

PAPER_COMPARISON = [
    ("MesoNet",      "WIFS 2018",   None,   83.10),
    ("Xception",     "ICCV 2019",   94.86,  92.39),
    ("F3-Net",       "ECCV 2020",   97.80,  93.12),
    ("M2TR",         "ICMR 2022",   96.75,  91.86),
    ("SPSL",         "CVPR 2021",   98.68,  95.51),
    ("MAT",          "CVPR 2021",   98.87,  96.61),
    ("HFI-Net",      "TIFS 2022",   97.07,  91.87),
    ("GocNet",       "ESWA 2023",   97.47,  93.48),
    ("HIFE",         "ESWA 2024",   98.83,  95.66),
    ("MH-FFNet",     "ESWA 2025",   99.44,  97.37),
]


def is_weights_healthy(sd):
    for name, t in sd.items():
        if not torch.isfinite(t).all():
            return False, name
    return True, None


def resolve_model_path():
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


def extract_state_dict(loaded):
    if isinstance(loaded, dict):
        if "state_dict" in loaded:
            return loaded["state_dict"]
        if "model_state_dict" in loaded:
            return loaded["model_state_dict"]
    return loaded


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


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


def find_optimal_threshold(labels, probs, lo=0.05, hi=0.95, step=0.01):
    from sklearn.metrics import f1_score as sk_f1
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(lo, hi, step):
        p = (np.asarray(probs) >= t).astype(float)
        f = sk_f1(labels, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 2), float(best_f1)


def ensure_val_cache(dataset, indices, num_workers=0):
    """Ensure validation samples are cached before running evaluation."""
    if dataset.cache_dir is None:
        print("Cache directory is not configured; skipping cache build.")
        return

    missing = [idx for idx in indices if not dataset.get_cache_path(idx).exists()]
    if not missing:
        print(f"Cache check passed: all {len(indices)} validation samples already cached.")
        return

    print(f"Cache missing for {len(missing)}/{len(indices)} validation samples.")
    print("Building cache for validation dataset before evaluation...")
    stats = dataset.build_cache(overwrite=False, indices=indices, num_workers=num_workers)
    print(
        f"Cache build finished: {stats['cached']} cached, "
        f"{stats['missing']} missing, cache_dir={stats['cache_dir']}"
    )

DOMAIN_DISPLAY = {
    "Deepfakes":         "Deepfakes",
    "Face2Face":         "Face2Face",
    "FaceSwap":          "FaceSwap",
    "FaceShifter":       "FaceShifter",
    "NeuralTextures":    "NeuralTextures",
    "DeepFakeDetection": "DeepFakeDetection",
    "youtube":           "YouTube (real)",
    "actors":            "Actors (real)",
}


def print_overall(metrics, model_path, n, opt_t=None, opt_f1=None):
    name = os.path.basename(model_path)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        bar_w = int(v * 30) if not math.isnan(v) else 0
        bar   = "█" * bar_w + "░" * (30 - bar_w)
        print(f"  {k:<12}: {v*100:>6.2f}%  {bar}")
    print(f"{'='*60}")
    print(f"  Eval samples : {n}")
    if opt_t is not None:
        bias = "toward FAKE" if opt_t < 0.45 else "toward REAL" if opt_t > 0.55 else "balanced"
        print(f"  Optimal thr  : {opt_t:.2f}  (F1={opt_f1*100:.2f}%  — bias {bias})")
    print()


def print_domain_table(per_domain):
    cols    = ["Acc%", "Prec%", "Rec%", "F1%", "AUC%", "IoU%", "N"]
    headers = f"{'Domain':<22}" + "".join(f"{c:>8}" for c in cols)
    sep     = "─" * len(headers)

    print(f"\n{'Per-Domain Metrics':^{len(headers)}}")
    print(sep)
    print(headers)
    print(sep)

    for dom in sorted(per_domain.keys()):
        m    = per_domain[dom]
        disp = DOMAIN_DISPLAY.get(dom, dom)

        def fmt(key):
            v = m[key]
            return f"{v*100:>7.2f}" if not math.isnan(v) else "    N/A"

        row = (f"{disp:<22}"
               f"{fmt('Accuracy'):>8}"
               f"{fmt('Precision'):>8}"
               f"{fmt('Recall'):>8}"
               f"{fmt('F1'):>8}"
               f"{fmt('AUC'):>8}"
               f"{fmt('IoU'):>8}"
               f"{m['N']:>6}")
        print(row)

    print(sep)


def print_paper_comparison(our_auc, our_acc):
    """Print comparison table """
    print(f"\n{'='*72}")
    print(f"  Comparison with Published Methods on FF++ (C23)")
    print(f"  Reference: Zhou et al., Expert Systems with Applications (2025)")
    print(f"{'='*72}")
    print(f"  {'Method':<16} {'Venue':<12} {'AUC%':>7} {'ACC%':>7}")
    print(f"  {'─'*16} {'─'*12} {'─'*7} {'─'*7}")

    rows = PAPER_COMPARISON + [("AFAGNetV3 (Ours)", "—", our_auc, our_acc)]
    best_auc = max(r[2] for r in rows if r[2] is not None)
    best_acc = max(r[3] for r in rows if r[3] is not None)

    for method, venue, auc_v, acc_v in rows:
        auc_str = f"{auc_v:>7.2f}" if auc_v else "    —  "
        acc_str = f"{acc_v:>7.2f}"
        auc_tag = " ◄" if auc_v == best_auc else "  "
        acc_tag = " ◄" if acc_v == best_acc else "  "
        is_ours = "Ours" in method
        print(f"  {'▶ ' if is_ours else '  '}{method:<16} {venue:<12}"
              f" {auc_str}{auc_tag} {acc_str}{acc_tag}")

    print(f"{'='*72}")
    print(f"  ◄ = best in column.  Our AUC gap to SOTA: "
          f"{best_auc - our_auc:+.2f}%  "
          f"(trained on RTX 3050 4GB, batch=2)")
    print()


def main():
    video_paths, labels, mask_paths, domains = build_ffpp_dataset(ROOT)

    dataset = FFPPDatasetV2(
        video_paths=video_paths, labels=labels,
        mask_paths=mask_paths, domains=domains,
        cache_dir=CACHE_DIR, use_alignment=True,
        training_mode=False, label_smoothing=0.0,
        use_sbi=False, temporal_strategy="blend",
    )

    _, val_indices = build_identity_disjoint_split(
        video_paths, val_ratio=0.2, seed=42
    )

    ensure_val_cache(dataset, val_indices, num_workers=0)

    eval_dataset = Subset(dataset, val_indices)
    loader = DataLoader(eval_dataset, batch_size=2, shuffle=False,
                        num_workers=0, pin_memory=False)

    model_path = resolve_model_path()
    model = AFAGNetV3().to(DEVICE)

    loaded = torch.load(model_path, map_location=DEVICE)
    state_dict = extract_state_dict(loaded)
    state_dict = strip_module_prefix(state_dict)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("WARNING: strict state_dict load failed. Trying to load compatible parameter subsets...")
        compatible, skipped, missing = load_state_dict_safe(model, state_dict)
        print(f"  Loaded {len(compatible)}/{len(model.state_dict())} compatible parameters.")
        print(f"  Skipped {len(skipped)} checkpoint keys due to mismatch or unexpected names.")
        print(f"  {len(missing)} model parameters were not initialized from checkpoint.")

    model.eval()

    print(f"\nCheckpoint : {model_path}")
    print(f"Eval size  : {len(eval_dataset)}")
    print(f"Device     : {DEVICE}")

    metrics, details = evaluate(
        model, loader, device=DEVICE,
        return_details=True, threshold=0.5
    )

    opt_t, opt_f1 = find_optimal_threshold(details["labels"], details["probs"])

    if abs(opt_t - 0.5) > 0.03:
        metrics_opt, details_opt = evaluate(
            model, loader, device=DEVICE,
            return_details=True, threshold=opt_t
        )
        print(f"\n  NOTE: Optimal threshold ({opt_t:.2f}) differs from 0.5.")
        print(f"  Showing per-domain table at optimal threshold.")
        per_domain   = details_opt["per_domain"]
        dom_labels   = details_opt["dom_labels"]
        dom_probs    = details_opt["dom_probs"]
        eval_metrics = metrics_opt
    else:
        per_domain   = details["per_domain"]
        dom_labels   = details["dom_labels"]
        dom_probs    = details["dom_probs"]
        eval_metrics = metrics

    print_overall(eval_metrics, model_path, len(eval_dataset), opt_t, opt_f1)
    print_domain_table(per_domain)
    print_paper_comparison(
        our_auc=eval_metrics.get("AUC", 0) * 100,
        our_acc=eval_metrics.get("Accuracy", 0) * 100,
    )

    ckpt_name = os.path.splitext(os.path.basename(model_path))[0]
    save_dir  = os.path.join("eval", ckpt_name)

    saved = save_evaluation_visuals(
        metrics     = eval_metrics,
        labels      = details["labels"],
        preds       = (np.asarray(details["probs"]) >= opt_t).astype(float),
        probs       = details["probs"],
        save_dir    = save_dir,
        prefix      = f"afagnet_v3_{ckpt_name}",
        per_domain  = per_domain,
        dom_labels  = dom_labels,
        dom_probs   = dom_probs,
        model_name  = "AFAGNetV3",
    )

    print(f"{'─'*60}")
    print(f"  Saved evaluation charts → {save_dir}/")
    for name, path in saved.items():
        print(f"    {name:<22}: {os.path.basename(path)}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
