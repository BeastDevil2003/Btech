import torch
import torch.nn as nn
from model.spatial import SpatialStream
from model.classifier import ClassifierHead


class SpatialOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialStream()
        self.classifier = ClassifierHead(in_channels=576)

    def forward(self, x):
        feat = self.spatial(x)
        return self.classifier(feat)
