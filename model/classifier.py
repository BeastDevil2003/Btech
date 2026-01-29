import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Lightweight classification head
    """

    def __init__(self, in_channels=576, num_classes=2, dropout=0.5):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, 7, 7]

        Returns:
            logits [B, num_classes]
        """
        x = self.gap(x)
        logits = self.classifier(x)
        return logits


