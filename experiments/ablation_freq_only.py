import torch
import torch.nn as nn
from model.frequency import FrequencyStream
from model.classifier import ClassifierHead


class FrequencyOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq = FrequencyStream()
        self.classifier = ClassifierHead(in_channels=576)

    def forward(self, x):
        feat = self.freq(x)
        return self.classifier(feat)
