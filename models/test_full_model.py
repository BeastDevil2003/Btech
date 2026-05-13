import torch
from models.afag_net import AFAGNet

B, N = 2, 16
x = torch.randn(B, N, 6, 224, 224)

model = AFAGNet()

out = model(x)

print("Cls:", out["pred_cls"].shape)
print("Mask:", out["pred_mask"].shape)
print("Csf:", out["Csf"].shape)