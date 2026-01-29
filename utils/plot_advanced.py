import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
import os

def plot_roc_pr_cm(y_true, y_probs, y_pred, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.savefig(f"{save_dir}/pr_curve.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()
