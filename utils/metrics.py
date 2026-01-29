from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }
