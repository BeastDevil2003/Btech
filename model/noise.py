import torch
import torch.nn as nn
import torchvision.models as models


class NoiseInjection(nn.Module):
    def __init__(self, std=0.03):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

        

