import torch
from torch.utils.data import DataLoader
from eval.evaluate import evaluate
from models.afag_net_v3 import AFAGNetV3


def run_cross_dataset(model_path, datasets_dict, device="cuda"):
    model = AFAGNetV3().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    results = {}

    for name, dataset in datasets_dict.items():
        print(f"\n[INFO] Evaluating on {name}")

        loader = DataLoader(dataset, batch_size=2)

        metrics = evaluate(model, loader, device)

        results[name] = metrics

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    return results
