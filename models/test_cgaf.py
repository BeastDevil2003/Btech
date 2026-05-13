import torch
from models.fusion.cgaf import CGAF

Fs = torch.randn(32, 256, 14, 14)
Ff = torch.randn(32, 256, 14, 14)
Fn = torch.randn(32, 256, 14, 14)

model = CGAF()

Ffusion, Csf, Csn = model(Fs, Ff, Fn)

print("Ffusion:", Ffusion.shape)
print("Csf:", Csf.shape)
print("Csn:", Csn.shape)