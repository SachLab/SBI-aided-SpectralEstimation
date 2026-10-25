import torch
import torch.nn.functional as F

from dataset import make_freq_grid


# build the grid-level label
def build_anchor_targets(freqs_hz, n_freq, amps, grid_hz, obj_sigma_bins=1.0):
    device = grid_hz.device
    G = grid_hz.shape[0]
    gs = float(grid_hz[1] - grid_hz[0])
    f = freqs_hz.to(device)                                    # [B,K]
    a = amps.to(device)
    B, K = f.shape
    valid = (torch.arange(K, device=device)[None, :] < n_freq.to(device)[:, None])  # [B,K]

    d = (grid_hz[None, None, :] - f[:, :, None]).abs()         # [B,K,G]
    argmax_bin = d.argmin(dim=-1)                              # [B,K]
    bin_center = grid_hz[argmax_bin]                           # [B,K]
    delta_true = ((f - bin_center) / gs).clamp(-0.5, 0.5)      # [B,K] 
    argmax_bin = torch.where(valid, argmax_bin, torch.full_like(argmax_bin, -1))
    delta_true = torch.where(valid, delta_true, torch.zeros_like(delta_true))

    bins = torch.arange(G, device=device).float()             # [G]
    ab = argmax_bin.clamp(min=0).float()                      # [B,K]
    if obj_sigma_bins > 0:
        g = torch.exp(-((bins[None, None, :] - ab[:, :, None]) ** 2) / (2 * obj_sigma_bins ** 2))
    else:
        g = (bins[None, None, :] == ab[:, :, None]).float()
    g = g * valid[:, :, None].float()                         # [B,K,G]
    obj_soft = g.max(dim=1).values                            # [B,G]

    return {"obj_soft": obj_soft, "argmax_bin": argmax_bin,
            "delta_true": delta_true, "valid": valid}


def anchor_loss(out, tgt, stage="stage1"):
    obj_logit, delta, log_sigma = out["obj_logit"], out["delta"], out["log_sigma"]
    vb = tgt["valid"]
    ab = tgt["argmax_bin"].clamp(min=0)                        # [B,K]
    nvalid = vb.float().sum().clamp_min(1.0)

    delta_g = torch.gather(delta, 1, ab)                       
    err2 = (delta_g - tgt["delta_true"]) ** 2                  # [B,K] 

    if stage == "stage1":
        l_obj = F.binary_cross_entropy_with_logits(obj_logit, tgt["obj_soft"])
        l_delta = (err2 * vb.float()).sum() / nvalid
        return l_obj + l_delta, {"l_obj": float(l_obj.detach()),
                                 "l_delta": float(l_delta.detach())}

    s_g = torch.gather(log_sigma, 1, ab)                       
    gnll = s_g + err2 / (2 * torch.exp(2 * s_g))               # [B,K]
    l_reg = (gnll * vb.float()).sum() / nvalid #l_reg=l_gnll=l_sigma

    return l_reg, {"l_reg": float(l_reg.detach())}

class AggregatorLoss:
    def __init__(self, cfg, grid, device="cpu"):
        acfg = cfg["encoder"]["anchor"]
        tcfg = cfg["crosswin"]["train"]
        self.grid = grid
        self.device = device
        _gs = float(cfg["frequency"]["grid_step_hz"])
        _sig_hz = acfg["obj_sigma_hz"]
        self.obj_sigma_bins = (float(_sig_hz) / _gs if _sig_hz is not None
                               else float(acfg["obj_sigma_bins"]))
        self.beta_gnll = float(tcfg["beta_gnll"])

    def __call__(self, obj_t, delta_t, log_sig_t, batch):
        B, M, G = obj_t.shape
        dev = self.device
        fw = batch["f_win"].to(dev).reshape(B * M, -1)                    
        amp = batch["amp_true"].to(dev)[:, None, :].expand(B, M, -1).reshape(B * M, -1)
        nf = batch["n_freq"].to(dev).repeat_interleave(M)
        tgt = build_anchor_targets(fw, nf, amp, self.grid,
                                   obj_sigma_bins=self.obj_sigma_bins)
        out = {"obj_logit": obj_t.reshape(B * M, G),
               "delta": delta_t.reshape(B * M, G),
               "log_sigma": log_sig_t.reshape(B * M, G)}

        l_anchor, sub = anchor_loss(out, tgt, stage="stage1")
        l_gnll, _ = anchor_loss(out, tgt, stage="stage2")

        total = l_anchor + self.beta_gnll * l_gnll
        parts = {**sub, "gnll": float(l_gnll.detach())}
        return total, parts


def build_aggregator_loss(cfg, grid, device="cpu"):
    return AggregatorLoss(cfg, grid, device=device)
