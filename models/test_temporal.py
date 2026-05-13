import torch
from models.temporal.temporal_model import TemporalModel

B, N = 2, 16
x = torch.randn(B, N, 256, 14, 14)

model = TemporalModel()

out = model(x)

print(out.shape)



