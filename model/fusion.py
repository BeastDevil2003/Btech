import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, channels=576, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.freq_to_spatial = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.spatial_to_freq = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, Fs, Ff):
        As = self.avg_pool(Fs)  
        Af = self.avg_pool(Ff)  


        Ws = self.sigmoid(self.freq_to_spatial(Af))   
        Wf = self.sigmoid(self.spatial_to_freq(As))   

        Fs_att = Fs * Ws
        Ff_att = Ff * Wf

        Fused = Fs_att + Ff_att

        return Fused, Ws, Wf






