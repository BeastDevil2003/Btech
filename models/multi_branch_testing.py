from models.multi_branch import MultiBranchModule
import torch

x = torch.randn(32, 256, 14, 14)

model = MultiBranchModule()

Fs, Ff, Fn = model(x)

print(Fs.shape, Ff.shape, Fn.shape)