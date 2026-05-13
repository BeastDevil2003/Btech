import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalModel(nn.Module):
    def __init__(self, in_channels=256, embed_dim=512, 
                 num_heads=8, num_layers=4, max_len=512):
        super().__init__()

        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)


        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w4 = nn.Parameter(torch.tensor(1.0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def get_spatial_mask(self, x):

        mask = torch.mean(torch.abs(x), dim=2, keepdim=True)
        
        mask_flattened = mask.view(mask.size(0), mask.size(1), -1)
        m_min = mask_flattened.min(dim=-1, keepdim=True)[0].view(-1, mask.size(1), 1, 1, 1)
        m_max = mask_flattened.max(dim=-1, keepdim=True)[0].view(-1, mask.size(1), 1, 1, 1)
        
        mask = (mask - m_min) / (m_max - m_min + 1e-6)
        return mask

    def global_pool_masked(self, x, mask):
        return (x * mask).sum(dim=[3, 4]) / (mask.sum(dim=[3, 4]) + 1e-6)

    def pad_time(self, x, target_len):
        pad_len = target_len - x.shape[1]
        if pad_len > 0:
            if x.shape[1] == 0:
                pad = x.new_zeros(x.shape[0], pad_len, x.shape[2])
            else:
                pad = x[:, -1:].repeat(1, pad_len, 1)
            x = torch.cat([x, pad], dim=1)
        return x

    def compute_differences(self, Ft):
        N = Ft.shape[1]

        D1 = torch.abs(Ft[:, 1:] - Ft[:, :-1])
        D2 = torch.abs(Ft[:, 2:] - Ft[:, :-2])
        D4 = torch.abs(Ft[:, 4:] - Ft[:, :-4])

        D1 = self.pad_time(D1, N)
        D2 = self.pad_time(D2, N)
        D4 = self.pad_time(D4, N)

        return D1, D2, D4

    def forward(self, Ffusion):
        B, N, C, H, W = Ffusion.shape

        mask = self.get_spatial_mask(Ffusion)
        Ft = self.global_pool_masked(Ffusion, mask)  # (B, N, C)

        D1, D2, D4 = self.compute_differences(Ft)

        sequence = torch.cat([
            Ft,
            self.w1 * 0.5 * D1,
            self.w2 * 0.3 * D2,
            self.w4 * 0.2 * D4
        ], dim=1)  # (B, 4N, C)

        sequence = self.proj(sequence)

        T = sequence.shape[1]
        sequence = sequence + self.pos_embed[:, :T, :]

        importance = torch.mean(torch.abs(D1), dim=-1, keepdim=True)  # (B, N, 1)
        importance_full = importance.repeat(1, 4, 1) # (B, 4N, 1)
        sequence = sequence + importance_full[:, :T, :]
        sequence = self.norm(sequence)
        out = self.transformer(sequence)
        video_feat = out.mean(dim=1)  

        return video_feat
