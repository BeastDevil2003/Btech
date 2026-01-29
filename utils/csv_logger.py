import csv
import os

class CSVLogger:
    def __init__(self, path="outputs/metrics.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_f1"
        ])

    def log(self, epoch, train_loss, train_acc, val_loss, val_f1):
        self.writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_acc:.4f}",
            f"{val_loss:.4f}",
            f"{val_f1:.4f}"
        ])
        self.file.flush()

    def close(self):
        self.file.close()
