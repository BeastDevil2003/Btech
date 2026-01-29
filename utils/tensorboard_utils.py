from torch.utils.tensorboard import SummaryWriter
import os

def get_writer(log_dir="outputs/tensorboard"):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)
