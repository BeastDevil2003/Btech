import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from model.model import DeepFakeDetectionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
    }

    return metrics, y_true, y_pred, y_prob


if __name__ == "__main__":
    from dataloader import val_loader

    model = DeepFakeDetectionModel().to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

    metrics, _, _, _ = evaluate(model, val_loader)

    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
