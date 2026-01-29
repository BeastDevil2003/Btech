import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_DIR = "experiments/results"
SAVE_DIR = "outputs/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

experiments = {
    "Full Model": "full.csv",
    "Spatial Only": "spatial_only.csv",
    "Frequency Only": "freq_only.csv",
    "No Attention": "no_attention.csv"
}

f1_scores = []
labels = []

for name, file in experiments.items():
    path = os.path.join(RESULT_DIR, file)
    df = pd.read_csv(path)
    best_f1 = df["val_f1"].max()
    f1_scores.append(best_f1)
    labels.append(name)

# Bar plot
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, f1_scores)

plt.ylabel("Best Validation F1-score")
plt.title("Ablation Study Comparison")
plt.ylim(0, 1)

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             y + 0.01,
             f"{y:.3f}",
             ha="center")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/ablation_f1_comparison.png")
plt.close()
