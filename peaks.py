import torch
import torch.nn.functional as F


def local_maxima_mask(values, min_distance):
    win = 2 * int(min_distance) + 1
    pooled = F.max_pool1d(values[:, None, :], kernel_size=win, stride=1,
                          padding=win // 2)[:, 0, :]
    is_max = values >= pooled
    left = F.pad(values[:, :-1], (1, 0), value=-float("inf"))
    return is_max & (values > left)


def topl_peak_indices(values, L, min_distance):
    B, G = values.shape
    mask = local_maxima_mask(values, min_distance)
    masked = torch.where(mask, values, torch.full_like(values, -float("inf")))
    L_eff = min(L, G)
    idx = torch.topk(masked, k=L_eff, dim=-1).indices
    fallback = values.argmax(dim=-1, keepdim=True).expand(B, L_eff)
    valid = torch.gather(masked, 1, idx) > -float("inf")
    idx = torch.where(valid, idx, fallback)
    if L_eff < L:
        idx = torch.cat([idx, fallback[:, :1].expand(B, L - L_eff)], dim=1)
    return idx
