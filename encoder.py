import math

import torch
import torch.nn as nn

from config import grid_size, signal_dim as _signal_dim
from dataset import film_input, film_input_dim, rate_ref_hz


class FiLM(nn.Module):
    def __init__(self, summary_dim, n_channels, hidden=64):
        super().__init__()
        self.C = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(summary_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * n_channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, summary):                 # x:[B,C,L], summary:[B,S]
        gb = self.mlp(summary)                      # [B,2C]
        gamma, beta = gb[:, :self.C], gb[:, self.C:]
        return (1.0 + gamma[..., None]) * x + beta[..., None]


class _FreqSelfAttnLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, d_model))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class NUDFTFreqTransformerAnchor(nn.Module):
    def __init__(self, signal_dim, grid_size, f_lo=0.0, f_hi=500.0,
                 d_model=128, n_heads=8, n_layers=4, ffn_dim=256,
                 dropout=0.0, use_film=True, summary_dim=4):
        super().__init__()
        self.grid_size = grid_size
        self.signal_dim = signal_dim
        self.register_buffer("freqs", torch.linspace(f_lo, f_hi, grid_size))
        self.in_proj = nn.Linear(2, d_model)                 # [Re,Im] -> d
        self.freq_embed = nn.Parameter(torch.randn(grid_size, d_model) * 0.02)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_film = bool(use_film)
        self.film = FiLM(summary_dim, d_model) if self.use_film else None
        self.layers = nn.ModuleList([_FreqSelfAttnLayer(d_model, n_heads, ffn_dim)
                                     for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.obj_head = nn.Linear(d_model, 1)
        self.delta_head = nn.Linear(d_model, 1)
        self.sigma_head = nn.Linear(d_model, 1)

    def forward(self, signal, t_n=None, rate_summary=None):
        b = signal.size(0)
        yr, yi = signal[:, 0], signal[:, 1]                  # [B,N]
        if t_n is None:
            t_n = torch.arange(self.signal_dim, device=signal.device,
                               dtype=signal.dtype)[None, :].expand(b, -1)
        phase = 2 * math.pi * self.freqs[None, :, None] * t_n[:, None, :]    # [B,G,N]
        cos, sin = torch.cos(phase), torch.sin(phase)
        sr = (yr[:, None, :] * cos + yi[:, None, :] * sin).sum(-1)           # [B,G]
        si = (yi[:, None, :] * cos - yr[:, None, :] * sin).sum(-1)           # [B,G]
        s = torch.stack([sr, si], dim=-1) / math.sqrt(self.signal_dim)       # [B,G,2] 
        x = self.in_proj(s) + self.freq_embed[None]                          # [B,G,d]
        if self.use_film and rate_summary is not None:
            x = self.film(x.transpose(1, 2), rate_summary).transpose(1, 2)   # [B,G,d]
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)                                     # self-attention
        x = self.final_norm(x)                               # [B,G,d]
        obj_logit = self.obj_head(x).squeeze(-1)             # [B,G]
        delta = 0.5 * torch.tanh(self.delta_head(x).squeeze(-1))
        log_sigma = self.sigma_head(x).squeeze(-1)
        return {"obj_logit": obj_logit, "delta": delta, "log_sigma": log_sigma}


def build_encoder(cfg):
    fcfg = cfg["frequency"]
    acfg = cfg["encoder"]["anchor"]
    t = cfg["encoder"]["ftf"]
    frr = cfg["data"]["f_rot_range"]
    if frr is not None:
        ref, mid = rate_ref_hz(cfg), 0.5 * (frr[0] + frr[1])
        if abs(ref - mid) > 0.15 * max(mid, 1e-6):
            raise ValueError(
                    f"data.rate_ref_hz={ref} differs too much from the midpoint {mid} of f_rot_range={frr}. "
                    f"It is the normalization reference for abs_mean and must be approximately equal to the midpoint; "
                    f"otherwise, the FiLM input scale will be inconsistent.")
        
    return NUDFTFreqTransformerAnchor(
        signal_dim=_signal_dim(cfg), grid_size=grid_size(cfg),
        f_lo=fcfg["f_min"], f_hi=fcfg["f_max"],
        d_model=int(t["d_model"]), n_heads=int(t["n_heads"]),
        n_layers=int(t["n_layers"]), ffn_dim=int(t["ffn_dim"]),
        dropout=float(t["dropout"]),
        use_film=bool(cfg["encoder"]["rate_inject"]),
        summary_dim=film_input_dim(cfg))


_FILM_KEY = {"summary": "rate_summary",        # [B,4]      
             "per_rev": "rev_speeds_norm"}     # [B,n_rev]  


def encoder_forward(model, batch, cfg, device):
    signal = batch["signal"].to(device)
    t_n = batch["t_n"].to(device)
    rate = None
    if cfg["encoder"]["rate_inject"]:
        rate = batch[_FILM_KEY[film_input(cfg)]].to(device)
    return model(signal, t_n, rate_summary=rate)
