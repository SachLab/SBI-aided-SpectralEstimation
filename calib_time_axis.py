import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_ckpt_config
from dataset import make_freq_grid
from peaks import topl_peak_indices as _topl_peak_indices
from crosswin_data import CrossWinDataset, n_windows
from axial_vit import AxialViT
from encoder import build_encoder
from loss import anchor_loss, build_anchor_targets
from train_report import RunningMean, TrainReporter, sigma_calibration

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AXIAL_CK, CFG = sys.argv[1], sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-4

cfg = load_ckpt_config(CFG)
K = cfg["signal"]["num_components"]
ecfg = cfg["eval"]; acfg = cfg["encoder"]["anchor"]
grid = torch.as_tensor(make_freq_grid(cfg), dtype=torch.float32, device=DEV)
G = grid.numel(); Δ = float(grid[1] - grid[0]); M = n_windows(cfg)
basin = ecfg["basin_hz"]
min_dist = max(int(round(ecfg["nms_distance_hz"] / Δ)), 1)
MONITOR_SNR = float(cfg["crosswin"]["eval"]["monitor_snr_db"])
EVAL_SIZE = int(cfg["crosswin"]["eval"]["eval_size"])
SNR_BANDS = ecfg["eval_snr_db"]            
FINAL_DL = list(ecfg.get("final_recall_dL", [0]))  
_gs = float(cfg["frequency"]["grid_step_hz"])
_sig_hz = acfg["obj_sigma_hz"]
obj_sigma_bins = float(_sig_hz) / _gs if _sig_hz is not None else float(acfg["obj_sigma_bins"])

ftf = build_encoder(cfg).to(DEV)
model = AxialViT(ftf, cfg).to(DEV)
sd = torch.load(AXIAL_CK, map_location=DEV, weights_only=False)
model.load_state_dict(sd["model_state"] if "model_state" in sd else sd, strict=True)
for p in model.parameters():
    p.requires_grad_(False)
sig_params = list(model.sigma_head.parameters())
for p in sig_params:
    p.requires_grad_(True)
# print(f"[σ-calib axial] 只训 sigma_head ({sum(p.numel() for p in sig_params)} parameters) | "
#       f"lr={LR:g} | G={G} M={M} K={K}")

opt = torch.optim.AdamW(sig_params, lr=LR)

def anchor_targets(batch):
    fw = batch["f_win"].to(DEV).reshape(-1, batch["f_win"].shape[-1])
    B = batch["f_win"].shape[0]
    amp = batch["amp_true"].to(DEV)[:, None, :].expand(B, M, -1).reshape(fw.shape[0], -1)
    nf = batch["n_freq"].to(DEV).repeat_interleave(M)
    return build_anchor_targets(fw, nf, amp, grid, obj_sigma_bins=obj_sigma_bins)


def flat_out(out):
    return {"obj_logit": out["obj_logit"].reshape(-1, G),
            "delta": out["delta"].reshape(-1, G),
            "log_sigma": out["log_sigma"].reshape(-1, G)}


@torch.no_grad()
def _run(snr, n, want_loss, dls=(0,)):
    dls = sorted(set(int(x) for x in dls) | {0})
    L_MAX = K + max(dls)
    ev = CrossWinDataset(cfg, n, seed=999000, snr_db=snr)
    res = {(k, d): {"r": [], "a": []} for k in ["model", "sum"] for d in dls}
    sig_all, err_all, vl, vp = [], [], [], []
    for b in DataLoader(ev, batch_size=32, num_workers=0):
        out = model(b, DEV)
        obj0, obj_t = out["obj0"], out["obj_logit"]
        if want_loss:
            fo = flat_out(out)
            l, parts = anchor_loss(fo, anchor_targets(b), stage="stage2")   
            vl.append(float(l))
            vp.append({k: float(v) for k, v in (parts or {}).items()})
        specs = {"model": (torch.sigmoid(obj_t), out["delta"]),
                 "sum": (torch.sigmoid(obj0.sum(1, keepdim=True).expand_as(obj0)),
                         out["delta0"])}
        for k, (s, dl) in specs.items():
            flat = s.reshape(-1, G)
            peaks = _topl_peak_indices(flat, L_MAX, min_dist)
            mu_all = grid[None, :] + dl.reshape(-1, G) * Δ
            pf_all = torch.gather(mu_all, 1, peaks)
            ft = b["f_win"].reshape(-1, b["f_win"].shape[-1]).to(DEV)
            nf = b["n_freq"].to(DEV).repeat_interleave(M)
            valid = torch.arange(ft.shape[1], device=DEV)[None] < nf[:, None]
            for dL in dls:
                keep = torch.arange(L_MAX, device=DEV)[None] < (nf + dL)[:, None]
                pf = torch.where(keep, pf_all, torch.full_like(pf_all, float("inf")))
                d = (pf[:, None, :] - ft[:, :, None]).abs()
                cov = (d.min(-1).values <= basin) & valid
                res[(k, dL)]["r"].append(float(cov.float().sum() / valid.float().sum().clamp_min(1)))
                res[(k, dL)]["a"].append(float((cov | ~valid).all(1).float().mean()))
        tgt = anchor_targets(b); fo = flat_out(out)
        aidx = tgt["argmax_bin"].clamp(min=0); aval = tgt["valid"]; dstar = tgt["delta_true"]
        dl = fo["delta"].cpu().numpy(); ls = fo["log_sigma"].cpu().numpy()
        av = aval.cpu().numpy(); ai = aidx.cpu().numpy(); ds_ = dstar.cpu().numpy()
        for i in range(dl.shape[0]):
            for j in range(ai.shape[1]):
                if not bool(av[i, j]):
                    continue
                g = int(ai[i, j])
                err_all.append(abs(float(dl[i, g]) - float(ds_[i, j])) * Δ)
                sig_all.append(float(np.exp(ls[i, g])) * Δ)
    corr, ratio = sigma_calibration(sig_all, err_all)
    o = {"recall": float(np.mean(res[("model", 0)]["r"])),
         "allk": float(np.mean(res[("model", 0)]["a"])),
         "sum_recall": float(np.mean(res[("sum", 0)]["r"])),
         "sum_allk": float(np.mean(res[("sum", 0)]["a"])),
         "sigma_corr": corr, "sigma_ratio": ratio}
    for dL in dls:
        if dL:
            o[f"recall@K+{dL}"] = float(np.mean(res[("model", dL)]["r"]))
            o[f"allk@K+{dL}"] = float(np.mean(res[("model", dL)]["a"]))
            o[f"sum_recall@K+{dL}"] = float(np.mean(res[("sum", dL)]["r"]))
            o[f"sum_allk@K+{dL}"] = float(np.mean(res[("sum", dL)]["a"]))
    if want_loss:
        o["val_parts"] = {"total": float(np.mean(vl))}
        for k in (vp[0] if vp else {}):
            o["val_parts"][k] = float(np.mean([p[k] for p in vp]))
    return o


@torch.no_grad()
def evaluate(n=None):
    n = n or EVAL_SIZE
    model.eval()
    m = _run(MONITOR_SNR, n, want_loss=True)
    model.train()
    return m


@torch.no_grad()
def evaluate_per_snr(n=None):
    n = n or EVAL_SIZE
    model.eval()
    out = {}
    keys = ["recall", "allk", "sum_recall", "sum_allk", "sigma_corr", "sigma_ratio"]
    keys += [f"{m}@K+{d}" for d in FINAL_DL if d
             for m in ("recall", "allk", "sum_recall", "sum_allk")]
    for snr in SNR_BANDS:
        m = _run(float(snr), n, want_loss=False, dls=FINAL_DL)
        out[str(snr)] = {k: m[k] for k in keys}
    return out


OUT_DIR = os.path.join(cfg["checkpoint"]["dir"], "axial_sigcalib")
rep = TrainReporter(OUT_DIR, cfg, stage="time_axis_calibration", select="loss",
                    train_sigma=True, tb_name="stage4_sigma_calib",
                    recall_L=K, monitor_snr_db=MONITOR_SNR)

SNR = cfg["training"]["train_snr_choices"]
BS = cfg["crosswin"]["train"]["batch_size"]
ds = CrossWinDataset(cfg, size=BS * ITERS, seed=7, snr_db=SNR)
loader = DataLoader(ds, batch_size=BS, num_workers=0)
rep.header()

runner = RunningMean(window=200)
for it, batch in enumerate(loader, 1):
    out = model(batch, DEV)
    fo = flat_out(out)
    loss, _parts = anchor_loss(fo, anchor_targets(batch), stage="stage2")   # 纯 GNLL
    opt.zero_grad(); loss.backward()
    opt.step()
    tr_parts = runner.update({"total": float(loss.detach()),
                              **{k: float(v) for k, v in (_parts or {}).items()}})

    if it == 1 or it % 500 == 0:
        m = evaluate()
        val_parts = m.pop("val_parts")
        ck = {"model_state": model.state_dict(), "iter": it, "cfg": cfg,
              "recall": m["recall"], "sigma_corr": m["sigma_corr"],
              "sigma_ratio": m["sigma_ratio"], "val_loss": val_parts["total"]}
        rep.log_step(it, tr_parts, val_parts, m, {"sigma_head": LR}, ckpt=ck)
        rep.print_row(it, tr_parts, val_parts, m, LR)
    if it >= ITERS:
        break

print("\nEvaluating across configured SNR levels...")
per_snr = evaluate_per_snr()
rep.finish(per_snr)
