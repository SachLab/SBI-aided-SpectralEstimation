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
from axial_vit import load_axial_vit
from loss import build_aggregator_loss
from scheduler import make_scheduler
from train_report import RunningMean, TrainReporter

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FTF_CK, CFG = sys.argv[1], sys.argv[2]

cfg = load_ckpt_config(CFG)
K = cfg["signal"]["num_components"]
etype = cfg["encoder"]["type"]

grid = torch.as_tensor(make_freq_grid(cfg), dtype=torch.float32, device=DEV)
G = grid.numel(); Δ = float(grid[1] - grid[0]); M = n_windows(cfg)
basin = cfg["eval"]["basin_hz"]
min_dist = max(int(round(cfg["eval"]["nms_distance_hz"] / Δ)), 1)

_tc = cfg["crosswin"]["train"]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else int(_tc["iters"])
EVAL_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else int(cfg["crosswin"]["eval"]["eval_every"])

model = load_axial_vit(cfg, grid, Δ, FTF_CK, device=DEV)
n_freq = sum(p.numel() for p in model.freq_layers.parameters())
n_time = sum(p.numel() for p in (pp for tl in model.time_layers for pp in tl.parameters()))
# print(f"[axial] frequency axis (FTF layers)={n_freq/1e6:.3f}M "
#       f"time axis ({len(model.time_layers)} layers, per-bin)={n_time/1e6:.3f}M")
# print(f"enc={etype} | G={G} n_win={M} K={K} | time-axis tokens=M×G={M*G}")


def recall_allk(spec, f_win, n_freq_, delta=None, dls=(0,)):
    dls = sorted(set(int(x) for x in dls) | {0})
    L_MAX = K + max(dls)
    B, Mw, _ = spec.shape
    flat = spec.reshape(B * Mw, G)
    peaks = _topl_peak_indices(flat, L_MAX, min_dist)
    if delta is None:
        pf_all = grid[peaks]
    else:
        mu_all = grid[None, :] + delta.reshape(B * Mw, G) * Δ
        pf_all = torch.gather(mu_all, 1, peaks)
    ft = f_win.reshape(B * Mw, -1).to(DEV)
    nf = n_freq_.to(DEV).repeat_interleave(Mw)
    valid = torch.arange(ft.shape[1], device=DEV)[None] < nf[:, None]
    res = {}
    for dL in dls:
        keep = torch.arange(L_MAX, device=DEV)[None] < (nf + dL)[:, None]
        pf = torch.where(keep, pf_all, torch.full_like(pf_all, float("inf")))
        d = (pf[:, None, :] - ft[:, :, None]).abs()
        covered = (d.min(-1).values <= basin) & valid
        res[dL] = (float(covered.float().sum() / valid.float().sum().clamp_min(1)),
                   float((covered | ~valid).all(1).float().mean()))
    return res


_ecfg = cfg["crosswin"]["eval"]
MONITOR_SNR = float(_ecfg["monitor_snr_db"]); EVAL_SIZE = int(_ecfg["eval_size"])
SNR = cfg["training"]["train_snr_choices"]
SNR_BANDS = cfg["eval"]["eval_snr_db"]     #
FINAL_DL = list(cfg["eval"].get("final_recall_dL", [0]))  


@torch.no_grad()
def _run(snr, n, want_loss, dls=(0,)):
    dls = sorted(set(int(x) for x in dls) | {0})
    ev = CrossWinDataset(cfg, n, seed=999000, snr_db=snr)
    res = {(k, d): {"r": [], "a": []} for k in ["model", "sum"] for d in dls}
    vl, vp = [], []
    for b in DataLoader(ev, batch_size=32, num_workers=0):
        out = model(b, DEV)
        obj0, obj_t = out["obj0"], out["obj_logit"]
        if want_loss:
            loss, parts = loss_fn(out["obj_logit"], out["delta"], out["log_sigma"], b)
            vl.append(float(loss))
            vp.append(parts)
        specs = {"model": (torch.sigmoid(obj_t), out["delta"]),                   
                 "sum": (torch.sigmoid(obj0.sum(1, keepdim=True).expand_as(obj0)),
                         out["delta0"])}                                          
        for k, (s, dl) in specs.items():
            for d, (r, a) in recall_allk(s, b["f_win"], b["n_freq"], dl, dls=dls).items():
                res[(k, d)]["r"].append(r); res[(k, d)]["a"].append(a)
    o = {"recall": float(np.mean(res[("model", 0)]["r"])),
         "allk": float(np.mean(res[("model", 0)]["a"])),
         "sum_recall": float(np.mean(res[("sum", 0)]["r"])),
         "sum_allk": float(np.mean(res[("sum", 0)]["a"]))}
    for d in dls:
        if d:
            o[f"recall@K+{d}"] = float(np.mean(res[("model", d)]["r"]))
            o[f"allk@K+{d}"] = float(np.mean(res[("model", d)]["a"]))
            o[f"sum_recall@K+{d}"] = float(np.mean(res[("sum", d)]["r"]))
            o[f"sum_allk@K+{d}"] = float(np.mean(res[("sum", d)]["a"]))
    if want_loss:
        o["val_parts"] = {"total": float(np.mean(vl))}
        for k in vp[0]:
            if k not in ACTIVE_PARTS:            
                continue
            o["val_parts"][k] = float(np.mean([p[k] for p in vp]))
    return o


@torch.no_grad()
def evaluate(n=None):
    n = n or EVAL_SIZE
    m = _run(MONITOR_SNR, n, want_loss=False)
    lo = _run(SNR, n, want_loss=True)
    m["val_parts"] = lo["val_parts"]
    return m


@torch.no_grad()
def evaluate_per_snr(n=None):
    n = n or EVAL_SIZE
    out = {}
    keys = ["recall", "allk", "sum_recall", "sum_allk"]
    keys += [f"{m}@K+{d}" for d in FINAL_DL if d
             for m in ("recall", "allk", "sum_recall", "sum_allk")]
    for snr in SNR_BANDS:
        m = _run(float(snr), n, want_loss=False, dls=FINAL_DL)
        out[str(snr)] = {k: m[k] for k in keys}
    return out


loss_fn = build_aggregator_loss(cfg, grid, device=DEV)

_beta = float(_tc.get("beta_gnll", 0.0))
FREEZE_SIGMA = _beta <= 0

ACTIVE_PARTS = {"l_obj", "l_delta"}
if _beta > 0:
    ACTIVE_PARTS.add("gnll")

TRAIN_SIGMA = not FREEZE_SIGMA
lr_time = float(_tc["lr"])
lr_freq = float(_tc.get("lr_freq", lr_time * 0.1))

freq_params, frozen = [], []
freq_modules = [model.in_proj, model.freq_embed, model.film, model.freq_layers,
                model.final_norm, model.obj_head, model.delta_head]
seen = set()
for mod in freq_modules:
    if mod is None:
        continue
    it_ = mod.parameters() if hasattr(mod, "parameters") else [mod]
    for p in it_:
        if id(p) not in seen:
            seen.add(id(p)); freq_params.append(p)
for p in model.sigma_head.parameters():
    if FREEZE_SIGMA:
        p.requires_grad_(False); frozen.append(p)
    else:
        freq_params.append(p)
time_params = [p for tl in model.time_layers for p in tl.parameters()]

FREEZE_FREQ = os.environ.get("FREEZE_FREQ", "1") == "1"
if FREEZE_FREQ:
    for p in freq_params:
        p.requires_grad_(False)
    opt = torch.optim.AdamW([{"params": time_params, "lr": lr_time}])
    # print(f"[FREEZE_FREQ] frequency axis frozen ({sum(p.numel() for p in freq_params)/1e6:.3f}M), "
    #       f"training time axis only ({sum(p.numel() for p in time_params)/1e6:.3f}M) | lr={lr_time:g}")
else:
    opt = torch.optim.AdamW([
        {"params": time_params, "lr": lr_time},
        {"params": freq_params, "lr": lr_freq},
    ])
    print(f"[End-to-end] time axis={lr_time:g}, frequency axis={lr_freq:g} ")

_sch_cfg = {"iters": ITERS, "anneal_start": int(_tc.get("anneal_start", ITERS)),
            "anneal_min_factor": float(_tc.get("anneal_min_factor", 0.01))}
if _sch_cfg["anneal_start"] > ITERS:
    _sch_cfg["anneal_start"] = int(round(ITERS * 0.6))
sched = make_scheduler(opt, _sch_cfg)

RESUME = os.environ.get("RESUME")
DATA_SEED = 1
if RESUME:
    _rs = torch.load(RESUME, map_location=DEV, weights_only=False)
    model.load_state_dict(_rs["model_state"], strict=True)
    DATA_SEED = int(os.environ.get("RESUME_SEED", "2000000"))
    print(f"[RESUME] {RESUME} (iter={_rs.get('iter')} recall={_rs.get('recall')}) "
          f"| additional steps={ITERS}, data seed={DATA_SEED}")

OUT_DIR = os.path.join(cfg["checkpoint"]["dir"],
                       ("axial_vit_frozenfreq" if FREEZE_FREQ else "axial_vit")
                       + ("_resume" if RESUME else ""))


_ck_root = os.path.basename(os.path.normpath(cfg["checkpoint"]["dir"]))
TB_NAME = ("stage3_time_axis" if _ck_root == "artifacts"
           else f"stage3_time_axis_{_ck_root}") + ("_resume" if RESUME else "")

rep = TrainReporter(OUT_DIR, cfg, stage="time_axis", select="recall",
                    train_sigma=TRAIN_SIGMA, tb_name=TB_NAME,
                    recall_L=K, monitor_snr_db=MONITOR_SNR)

BS = _tc["batch_size"]
ds = CrossWinDataset(cfg, size=BS * ITERS, seed=DATA_SEED, snr_db=SNR)
loader = DataLoader(ds, batch_size=BS, num_workers=0)
d = cfg["data"]
_cpl = f" coupling={d.get('drift_coupling_range')}" if d["freq_drift"] else ""
# print(f"Data: regime={d['regime']} freq_drift={d['freq_drift']}{_cpl}"
#       f" | SNR~U[{SNR['min']},{SNR['max']}]")
# print(f"Checkpoint selection: highest val_recall@{K}@{MONITOR_SNR:g}dB "
#       "(model path; sum path is logged only)")
rep.header()

runner = RunningMean(window=200)
all_train = time_params if FREEZE_FREQ else (time_params + freq_params)
for it, batch in enumerate(loader, 1):
    out = model(batch, DEV)
    loss, parts = loss_fn(out["obj_logit"], out["delta"], out["log_sigma"], batch)
    opt.zero_grad(); loss.backward()
    opt.step(); sched.step()

    tr_parts = runner.update({"total": float(loss.detach()),
                              **{k: v for k, v in parts.items() if k in ACTIVE_PARTS}})

    if it == 1 or it % EVAL_EVERY == 0:
        model.eval(); m = evaluate(); model.train()
        cur_lr = opt.param_groups[0]["lr"]
        val_parts = m.pop("val_parts")
        ck = {"model_state": model.state_dict(), "iter": it, "cfg": cfg,
              "recall": m["recall"], "allk": m["allk"], "val_loss": val_parts["total"]}
        rep.log_step(it, tr_parts, val_parts, m, {"time_axis": cur_lr}, ckpt=ck)
        rep.print_row(it, tr_parts, val_parts, m, cur_lr)
    if it >= ITERS:
        break

print("\nEvaluating across configured SNR levels...")
model.eval()
per_snr = evaluate_per_snr()
rep.finish(per_snr)
