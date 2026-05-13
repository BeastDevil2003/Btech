import torch
from models.backbone import AFAGBackbone

model = AFAGBackbone()

x = torch.randn(32, 6, 224, 224)

F_low, F_high = model(x)

print("F_low:", F_low.shape)
print("F_high:", F_high.shape)