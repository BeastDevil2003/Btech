import os
import torch
import torch.nn as nn
import torch.optim as optim

from model.model import DeepFakeDetectionModel   
from utils.metrics import compute_metrics



from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler
from torch.amp import GradScaler
from tqdm.auto import tqdm


from dataloader import train_loader, val_loader

from utils.plot_metrics import plot_training_curves
from utils.logger import setup_logger
from utils.tensorboard_utils import get_writer
from utils.csv_logger import CSVLogger

logger = setup_logger()
csv_logger = CSVLogger()
writer = get_writer()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True    
print(" Training on device:", DEVICE)   
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
SAVE_PATH = "best_model.pth"


PATIENCE = 5   

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0   

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast():    
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0    
    preds_all, labels_all = [], []   
    all_labels = []     
    all_preds = []      
    all_probs = []       


    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())
        running_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)

    metrics = compute_metrics(preds_all, labels_all)
    metrics["loss"] = running_loss / len(loader)

    return metrics

def main():

    model = DeepFakeDetectionModel(num_classes=2).to(DEVICE) 
    print(" Model device:", next(model.parameters()).device)

    if torch.cuda.device_count() > 1:
        print(f" Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)   

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device='cuda')


    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]

        print(f" Resuming training from epoch {start_epoch}")
    else:
        print(" Starting fresh training")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_f1s = []
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        metrics = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        train_accs.append(metrics['accuracy'])
        val_losses.append(metrics['loss'])
        val_f1s.append(metrics['f1'])


        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {metrics['loss']:.4f}")
        print(f"Accuracy   : {metrics['accuracy']:.4f}")
        print(f"F1 Score   : {metrics['f1']:.4f}")

        logger.info(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {metrics['loss']:.4f} | "
        f"Val F1: {metrics['f1']:.4f}"
        )


        csv_logger.log(
        epoch,
        train_loss,
        metrics['accuracy'],
        metrics['loss'],
        metrics['f1']
        )


        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", metrics['loss'], epoch)
        writer.add_scalar("Accuracy/Train", metrics['accuracy'], epoch)
        writer.add_scalar("F1/Val", metrics['f1'], epoch)   

        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f1": best_f1
            }, SAVE_PATH)

            print(" Best model saved (by F1-score)")
        else:
            patience_counter += 1
            print(f" Early stopping patience: {patience_counter}/{PATIENCE}")

        if epoch in [1, 5, 10, 20]:
            from utils.attention_viz import save_attention_map
            save_attention_map(model, sample_img, epoch)

        if patience_counter >= PATIENCE:
            print(" Early stopping triggered")
            break
    

    csv_logger.close()
    writer.close()



    from utils.plot_metrics import plot_training_curves

    plot_training_curves(train_losses, val_losses, train_accs, val_f1s)

    from utils.plot_advanced import plot_roc_pr_cm
    plot_roc_pr_cm(all_labels,all_probs,all_preds)


"""
if __name__ == "__main__":
    from dataloader import train_loader, val_loader
    main(train_loader, val_loader)
"""

if __name__ == "__main__":
    main()
