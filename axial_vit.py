import math

import torch
from torch import nn

from encoder import NUDFTFreqTransformerAnchor

TWO_PI = 2.0 * math.pi


class _TimeAxisRoPEAttn(nn.Module):
    def __init__(self, d, nhead, fmin=0.1, fmax=10.0):
        super().__init__()
        self.d, self.h, self.dh = d, nhead, d // nhead
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        nn.init.zeros_(self.o.weight)          
        nn.init.zeros_(self.o.bias)
        nf = self.dh // 2
        freqs = torch.logspace(math.log10(fmin), math.log10(fmax), nf)
        self.register_buffer("rot_w", TWO_PI * freqs)

    def _rope(self, x, tau):
        # x[Bg,h,M,dh], tau[Bg,M] 
        Bg, h, M, dh = x.shape
        nf = dh // 2
        ang = self.rot_w.view(1, 1, 1, nf) * tau.view(Bg, 1, M, 1)
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    def forward(self, x, tau):
        # x[Bg,M,d], tau[Bg,M] → attention over M (
        Bg, M, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(Bg, M, self.h, self.dh).transpose(1, 2)
        k = k.view(Bg, M, self.h, self.dh).transpose(1, 2)
        v = v.view(Bg, M, self.h, self.dh).transpose(1, 2)
        q, k = self._rope(q, tau), self._rope(k, tau)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        o = (a.softmax(-1) @ v).transpose(1, 2).reshape(Bg, M, self.d)
        return self.o(o)


class _TimeAxisLayer(nn.Module):
    def __init__(self, d, nhead, ff):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = _TimeAxisRoPEAttn(d, nhead)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        nn.init.zeros_(self.ff[-1].weight)    
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x, tau):
        B, M, G, d = x.shape
        xr = x.permute(0, 2, 1, 3).reshape(B * G, M, d)     
        taur = tau[:, None, :].expand(B, G, M).reshape(B * G, M)   
        xr = xr + self.attn(self.n1(xr), taur)
        xr = xr + self.ff(self.n2(xr))
        return xr.reshape(B, G, M, d).permute(0, 2, 1, 3)  


class AxialViT(nn.Module):
    def __init__(self, ftf: NUDFTFreqTransformerAnchor, cfg, n_time_layers=None):
        super().__init__()
        self.cfg = cfg
        self.G = int(ftf.grid_size)
        self.signal_dim = int(ftf.signal_dim)
        self.register_buffer("freqs", ftf.freqs.clone())
        self.in_proj = ftf.in_proj
        self.freq_embed = ftf.freq_embed
        self.drop = ftf.drop
        self.use_film = ftf.use_film
        self.film = ftf.film
        self.freq_layers = ftf.layers                    # ModuleList[_FreqSelfAttnLayer] × L
        self.final_norm = ftf.final_norm
        self.obj_head, self.delta_head, self.sigma_head = ftf.obj_head, ftf.delta_head, ftf.sigma_head
        L = len(self.freq_layers)
        n_time = L if n_time_layers is None else int(n_time_layers)
        acfg = cfg["crosswin"]["time_axis"]
        d = self.freq_embed.shape[1]
        self.time_layers = nn.ModuleList([
            _TimeAxisLayer(d, int(acfg["n_heads"]), int(acfg["ffn_dim"]))
            for _ in range(n_time)])
        self.n_pair = min(L, n_time)

    @staticmethod
    def _apply_head(h, layer):
        return layer(h).squeeze(-1)                      # h[B,M,G,d] → [B,M,G]

    def _nudft(self, yr, yi, t_n):
        # [B·M,N] → S[B·M,G,2] → token[B·M,G,d]. 
        phase = 2 * math.pi * self.freqs[None, :, None] * t_n[:, None, :]
        cos, sin = torch.cos(phase), torch.sin(phase)
        sr = (yr[:, None, :] * cos + yi[:, None, :] * sin).sum(-1)
        si = (yi[:, None, :] * cos - yr[:, None, :] * sin).sum(-1)
        s = torch.stack([sr, si], dim=-1) / math.sqrt(self.signal_dim)
        return self.in_proj(s) + self.freq_embed[None]

    def _freq_tokens(self, batch, device):
        from dataset import film_input           
        from encoder import _FILM_KEY
        B, M = batch["signal"].shape[:2]
        sig = batch["signal"].to(device).reshape(B * M, 2, -1)
        t_n = batch["t_n"].to(device).reshape(B * M, -1)
        yr, yi = sig[:, 0], sig[:, 1]
        x = self._nudft(yr, yi, t_n)                     # [B·M,G,d]
        if self.use_film and self.cfg["encoder"]["rate_inject"]:
            rk = _FILM_KEY[film_input(self.cfg)]
            rate = batch[rk].to(device)
            rate = rate.reshape(B * M, *rate.shape[2:])
            x = self.film(x.transpose(1, 2), rate).transpose(1, 2)
        x = self.drop(x)
        return x.reshape(B, M, self.G, -1)

    def _heads(self, x_freq):
        obj = self._apply_head(x_freq, self.obj_head)
        delta = 0.5 * torch.tanh(self._apply_head(x_freq, self.delta_head))
        log_sigma = self._apply_head(x_freq, self.sigma_head)    
        return obj, delta, log_sigma

    def forward(self, batch, device):
        B, M = batch["signal"].shape[:2]
        tau = batch["tau"].to(device)                    
        x = self._freq_tokens(batch, device)             # [B,M,G,d]

        x0 = x
        for fl in self.freq_layers:
            xf = x0.reshape(B * M, self.G, -1)
            x0 = fl(xf).reshape(B, M, self.G, -1)
        x0 = self.final_norm(x0)
        obj0, delta0, log_sig0 = self._heads(x0)

        for i in range(len(self.freq_layers)):
            xf = x.reshape(B * M, self.G, -1)
            x = self.freq_layers[i](xf).reshape(B, M, self.G, -1)   
            if i < self.n_pair:
                x = self.time_layers[i](x, tau)                     
        x = self.final_norm(x)
        obj_t, delta_t, log_sig_t = self._heads(x)
        return {"obj_logit": obj_t, "delta": delta_t, "log_sigma": log_sig_t,
                "obj0": obj0, "delta0": delta0, "log_sig0": log_sig0,
                "feat": x}


def load_axial_vit(cfg, grid, grid_step, enc_ckpt, device="cpu", n_time_layers=None):
    from encoder import build_encoder
    ftf = build_encoder(cfg).to(device)
    sd = torch.load(enc_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    ftf.load_state_dict(sd, strict=True)                 # FTF 本体 strict
    model = AxialViT(ftf, cfg, n_time_layers=n_time_layers).to(device)
    return model
