import torch
import torch.nn as nn
import math

class BlockDCT(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        self.register_buffer("dct_matrix", self.create_dct_matrix(block_size))

    def create_dct_matrix(self, N):
        mat = torch.zeros((N, N))
        for k in range(N):
            for n in range(N):
                coef = math.sqrt(1 / N) if k == 0 else math.sqrt(2 / N)
                mat[k, n] = coef * math.cos((math.pi * (2 * n + 1) * k) / (2 * N))    
        return mat


    def forward(self, x):
        B, C, H, W = x.shape

        x = x.unfold(2, self.block_size, self.block_size)
        x = x.unfold(3, self.block_size, self.block_size)

        x = x.contiguous().view(-1, self.block_size, self.block_size)

        dct = self.dct_matrix @ x @ self.dct_matrix.t()
        dct = dct.view(
            B, C,
            H // self.block_size,
            W // self.block_size,
            self.block_size,
            self.block_size
        )

        return dct


class FrequencySelector(nn.Module):
    def __init__(self, start=4):
        super().__init__()
        self.start = start

    def forward(self, dct):
        return dct[:, :, :, :, self.start:, self.start:]


class FrequencyCNN(nn.Module):
    def __init__(self, out_channels=576):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class FrequencyStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.dct = BlockDCT()    
        self.selector = FrequencySelector()     
        self.proj = FrequencyCNN(out_channels=576)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        dct = self.dct(x)
        freq = self.selector(dct)

        B, C, h, w, bh, bw = freq.shape
        freq = freq.reshape(B, C * bh * bw, h, w)  

        freq = self.proj(freq)  
        freq = self.pool(freq) 

        return freq
    
