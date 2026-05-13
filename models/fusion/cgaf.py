import torch
import torch.nn as nn
import torch.nn.functional as F


class CGAFv2(nn.Module):
    def __init__(self, channels: int = 256, use_f_high: bool = True):
        super().__init__()

        self.channels    = channels
        self.use_f_high  = use_f_high

        self.proj_s = nn.Conv2d(channels, channels, 1)
        self.proj_f = nn.Conv2d(channels, channels, 1)
        self.proj_n = nn.Conv2d(channels, channels, 1)


        self.log_temp = nn.Parameter(torch.tensor(0.0))

        if use_f_high:
            self.high_gate = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.GELU(),
                nn.Linear(channels // 4, 1),
                nn.Sigmoid(),
            )

        self.alpha_raw = nn.Parameter(torch.tensor(0.693))   

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),   
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    @property
    def temp(self) -> torch.Tensor:
        """Temperature in [0.1, 10.0] via exponential parameterization."""
        return torch.clamp(torch.exp(self.log_temp), min=0.1, max=10.0)

    @property
    def alpha(self) -> torch.Tensor:
        """Learned residual weight, always positive via softplus."""
        return F.softplus(self.alpha_raw)

    def compute_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = F.normalize(A, dim=1)
        B = F.normalize(B, dim=1)
        return torch.sum(A * B, dim=1, keepdim=True)

    def forward(
        self,
        Fs: torch.Tensor,
        Ff: torch.Tensor,
        Fn: torch.Tensor,
        F_high: torch.Tensor = None,
    ):
     
        Fs_p = self.proj_s(Fs)
        Ff_p = self.proj_f(Ff)
        Fn_p = self.proj_n(Fn)

        Csf = self.compute_similarity(Fs_p, Ff_p)   
        Csn = self.compute_similarity(Fs_p, Fn_p)   
        Cfn = self.compute_similarity(Ff_p, Fn_p)  

        Wf = torch.sigmoid(Csf * self.temp)
        Wn = torch.sigmoid(Csn * self.temp)


        W_sum = Wf + Wn + 1e-6
        Wf = Wf / W_sum
        Wn = Wn / W_sum

        cross_gate = torch.sigmoid(Cfn)
        Wf = Wf * cross_gate
        Wn = Wn * cross_gate

        if self.use_f_high and F_high is not None:
            high_pooled = F.adaptive_avg_pool2d(F_high, 1).flatten(1)  
            global_gate = self.high_gate(high_pooled).view(-1, 1, 1, 1)  
            Wf = Wf * (0.5 + 0.5 * global_gate)
            Wn = Wn * (0.5 + 0.5 * global_gate)

        fused_aux = Wf * Ff + Wn * Fn
        Ffusion   = Fs + self.alpha * fused_aux

        Ffusion = self.refine(Ffusion)

        return Ffusion, Csf, Csn, Cfn


def consistency_loss(Fs, Ff, Fn):
    return F.l1_loss(Fs, Ff) + F.l1_loss(Fs, Fn)