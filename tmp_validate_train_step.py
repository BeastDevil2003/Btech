import torch
from torch.utils.data import Dataset, DataLoader
from models.afag_net_v3 import AFAGNetV3
from train.train_pipeline_v3 import train_one_epoch, FocalLoss
import torch.optim as optim

class DummyVideoDataset(Dataset):
    def __init__(self, length=4, num_frames=8):
        self.length = length
        self.num_frames = num_frames
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return {
            'frames': torch.randn(self.num_frames, 6, 224, 224),
            'label': torch.tensor(1.0 if idx % 2 else 0.0),
            'mask_frames': torch.randn(self.num_frames, 224, 224),
            'has_mask': True,
            'domain': 'dummy'
        }

model = AFAGNetV3()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
train_ds = DummyVideoDataset(length=4, num_frames=8)
loader = DataLoader(train_ds, batch_size=2, shuffle=False)
cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

result = train_one_epoch(
    model=model,
    loader=loader,
    optimizer=optimizer,
    device='cpu',
    scaler=torch.cuda.amp.GradScaler(enabled=False),
    epoch=0,
    total_epochs=1,
    use_amp=False,
    cls_loss_fn=cls_loss_fn,
    warmup_epochs=2,
    use_mixup=False,
    mixup_alpha=0.2,
    ema=None,
)
print(result)
