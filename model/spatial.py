import torch
import torch.nn as nn
import torchvision.models as models

class SpatialStream(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v3_small(pretrained=True)

        self.features = backbone.features

        self.out_channels = 576

    def forward(self, x):             
        x = self.features(x)          
        return x
