import torch
from models.heads.output_heads import OutputHeads

B, N = 2, 16

video_feat = torch.randn(B, 512)
Ffusion = torch.randn(B, N, 256, 14, 14)

model = OutputHeads()

cls, mask = model(video_feat, Ffusion)

print("Cls:", cls.shape)
print("Mask:", mask.shape)