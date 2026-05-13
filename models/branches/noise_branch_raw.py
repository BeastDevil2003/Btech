
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 3-KERNEL SRM — CURRENT 
# ===========================================================================

def get_srm_kernels():
    """
    3 representative SRM kernels:
      k1: first-order horizontal difference  (detects horizontal blending seams)
      k2: second-order horizontal (Laplacian row): detects noise level changes
      k3: 2D Laplacian: detects texture discontinuities at blending boundaries
    """
    k1 = [[0,  0,  0],
          [0,  1, -1],
          [0,  0,  0]]   # first-order horizontal

    k2 = [[0,  0,  0],
          [1, -2,  1],
          [0,  0,  0]]   # second-order horizontal

    k3 = [[-1,  2, -1],
          [ 2, -4,  2],
          [-1,  2, -1]]  # 2D Laplacian

    kernels = torch.tensor([k1, k2, k3], dtype=torch.float32)
    return kernels.unsqueeze(1)  




# def get_srm_kernels_full():
#     kernels = []
#
#     # First-order kernels (4 directions)
#     kernels.append([[0, 0, 0], [0,  1, -1], [0,  0,  0]])   # h  (horizontal)
#     kernels.append([[0, 0, 0], [0,  1,  0], [0, -1,  0]])   # v  (vertical)
#     kernels.append([[0, 0, 0], [0,  1,  0], [0,  0, -1]])   # d1 (diagonal)
#     kernels.append([[0, 0, 0], [0,  1,  0], [-1, 0,  0]])   # d2 (anti-diagonal)
#
#     # Second-order kernels (4 directions)
#     kernels.append([[0,  0, 0], [1, -2,  1], [0,  0,  0]])  # h2
#     kernels.append([[0,  1, 0], [0, -2,  0], [0,  1,  0]])  # v2
#     kernels.append([[1,  0, 0], [0, -2,  0], [0,  0,  1]])  # d3
#     kernels.append([[0,  0, 1], [0, -2,  0], [1,  0,  0]])  # d4
#
#     # 2D Laplacian (cross pattern)
#     kernels.append([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
#
#     # Third-order horizontal
#     kernels.append([[0, 0, 0], [-1, 3, -3], [0, 0, 0]])    # hmm left
#     kernels.append([[0, 0, 0], [1, -3,  3], [0, 0, 0]])    # hmm right
#
#     # Third-order vertical
#     kernels.append([[0, -1, 0], [0,  3, 0], [0, -3, 0]])
#     kernels.append([[0,  1, 0], [0, -3, 0], [0,  3, 0]])
#
#     # Square kernel patterns
#     kernels.append([[ 1, -2,  1], [-2,  4, -2], [ 1, -2,  1]])   # square2
#     kernels.append([[-1,  2, -1], [ 2, -4,  2], [-1,  2, -1]])   # square3
#
#     # Cross consistency kernels
#     kernels.append([[0,  1, 0], [1, -4,  1], [0,  1,  0]])   # cross4
#     kernels.append([[1,  0, 1], [0, -4,  0], [1,  0,  1]])   # cross5
#
#     # Higher-order diagonal patterns
#     kernels.append([[ 2, -1, 0], [-1,  0,  1], [0,   1, -2]])
#     kernels.append([[ 0, -1, 2], [ 1,  0, -1], [-2,  1,  0]])
#     kernels.append([[-2,  1, 0], [ 1,  0, -1], [0,  -1,  2]])
#     kernels.append([[ 0,  1,-2], [-1,  0,  1], [ 2, -1,  0]])
#
#     # Horizontal span-2 patterns
#     kernels.append([[0, 0, 0], [1, -1,  0], [0,  0,  0]])
#     kernels.append([[0, 0, 0], [0,  1, -1], [0,  0,  0]])
#
#     # Vertical span-2 patterns
#     kernels.append([[0, 1, 0], [0, -1, 0], [0,  0,  0]])
#     kernels.append([[0, 0, 0], [0,  1, 0], [0, -1,  0]])
#
#     # Edge enhancement patterns
#     kernels.append([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]])   # sharpen
#     kernels.append([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])   # strong sharpen
#
#     # Smoothness residual patterns
#     kernels.append([[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]])   # (normalize after)
#     kernels.append([[ 0,  0,  0], [ 0,  1,  0], [ 0,  0,  0]])   # identity residual
#
#     k = torch.tensor(kernels, dtype=torch.float32)
#     # Normalize each kernel to zero-sum (ensures noise residual extraction)
#     k = k - k.mean(dim=[1, 2], keepdim=True)
#     return k.unsqueeze(1)   # (30, 1, 3, 3)


# ===========================================================================
# NOISE BRANCH
# ===========================================================================

class NoiseBranch(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),    
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

        )


        srm_kernels = get_srm_kernels()  
        # srm_kernels = get_srm_kernels_full()              # for 30 kernels (T4)    uncommentthis line and whole code of30 kernel
        self.srm = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False, groups=out_channels
        )
        with torch.no_grad():
            k = srm_kernels.mean(dim=0)   
            self.srm.weight.copy_(k.repeat(out_channels, 1, 1, 1))
        for p in self.srm.parameters():
            p.requires_grad = False

        self.learnable_path = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, groups=out_channels, bias=False
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_adapted = self.adapter(x)    
        x_norm = F.instance_norm(x_adapted)

        noise = self.srm(x_norm) + self.learnable_path(x_norm)

        out  = self.conv(noise)
        attn = self.attn(out)

        return out * attn + x_adapted