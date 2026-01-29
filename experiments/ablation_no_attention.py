import torch
import torch.nn as nn
from model.spatial import SpatialStream
from model.frequency import FrequencyStream
from model.classifier import ClassifierHead


class NoAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialStream()
        self.freq = FrequencyStream()
        self.classifier = ClassifierHead(in_channels=1152)

    def forward(self, x):
        Fs = self.spatial(x)
        Ff = self.freq(x)
        fused = torch.cat([Fs, Ff], dim=1)
        return self.classifier(fused)

