import matplotlib.pyplot as plt
import os

def plot_training_curves(train_losses, val_losses,
                         train_accs, val_f1s,
                         save_dir="outputs/plots"):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()


    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_f1s, label="Val F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Accuracy & F1 vs Epoch")
    plt.savefig(f"{save_dir}/accuracy_f1_curve.png")
    plt.close()
