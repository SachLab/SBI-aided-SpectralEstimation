import json
import os
import time

import numpy as np
import torch

from ckpt_root import TB_ROOT, no_clobber


def dump_config(out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str, ensure_ascii=False)


class RunningMean:
    def __init__(self, window=200):
        self.window = int(window)
        self._buf = {}

    def update(self, parts):
        for k, v in parts.items():
            b = self._buf.setdefault(k, [])
            b.append(float(v))
            if len(b) > self.window:
                b.pop(0)
        return self.value()

    def value(self):
        return {k: float(np.mean(v)) for k, v in self._buf.items() if v}


def sigma_calibration(sigmas, errs):
    s = np.asarray(sigmas, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    ok = (s > 0) & np.isfinite(s) & np.isfinite(e)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    corr = float(np.corrcoef(s[ok], e[ok])[0, 1])
    ratio = float(np.median(s[ok] / np.clip(e[ok], 1e-6, None)))
    return corr, ratio


class TrainReporter:
    def __init__(self, out_dir, cfg, stage, select, train_sigma,
                 tb_name=None, tb_dir=None, recall_L=None, monitor_snr_db=20.0):
        assert select in ("recall", "loss"), f"selection criteria for 'best.pt' can only be chosen from recall/loss and produce {select!r}"
        self.out_dir = no_clobber(str(out_dir))
        self.cfg = cfg
        self.stage = stage
        self.select = select
        self.train_sigma = bool(train_sigma)
        self.recall_L = recall_L
        self.monitor_snr_db = float(monitor_snr_db)

        os.makedirs(self.out_dir, exist_ok=True)
        dump_config(self.out_dir, cfg)                    

        self.train_log = []
        self.best_recall = -float("inf")
        self.best_val = float("inf")
        self.t0 = time.time()

        self.tb = None
        _dir = tb_dir or cfg.get("tb_dir") or TB_ROOT
        if tb_name:
            tb_name = os.path.basename(no_clobber(os.path.join(_dir, tb_name)))
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(_dir, tb_name))
                print(f"[tb] {os.path.join(_dir, tb_name)}")
            except Exception as exc:                        
                print(f"[tb] is not opened ({exc})")

    def log_step(self, it, train_parts, val_parts, metrics, lrs, ckpt=None):
        row = {"iter": int(it), "elapsed_s": time.time() - self.t0}
        keep = set(self._loss_keys(train_parts, val_parts)) | {"total"}
        for k, v in train_parts.items():
            if k in keep:
                row[f"train_{k}"] = float(v)
        for k, v in val_parts.items():
            if k in keep:
                row[f"val_{k}"] = float(v)
        for k, v in metrics.items():
            row[k] = float(v)
        row["lr"] = {k: float(v) for k, v in lrs.items()}
        self.train_log.append(row)

        with open(os.path.join(self.out_dir, "train_log.json"), "w", encoding="utf-8") as fh:
            json.dump(self.train_log, fh, indent=2, ensure_ascii=False)

        if self.tb is not None:
            tr_t, va_t = train_parts.get("total"), val_parts.get("total")
            if tr_t is not None and va_t is not None:
                self.tb.add_scalars("loss/train_vs_val",
                                    {"train": float(tr_t), "val": float(va_t)}, it)
            for k, v in train_parts.items():
                if k in keep:
                    self.tb.add_scalar(f"loss_train/{k}", float(v), it)
            for k, v in val_parts.items():
                if k in keep:
                    self.tb.add_scalar(f"loss_val/{k}", float(v), it)
            for k, v in metrics.items():
                self.tb.add_scalar(f"metric/{k}", float(v), it)
            for k, v in lrs.items():
                self.tb.add_scalar(f"lr/{k}", float(v), it)
            self.tb.flush()

        if self.select == "recall":
            score, best = metrics.get("recall", float("nan")), self.best_recall
        else:
            score, best = -float(val_parts.get("total", float("inf"))), -self.best_val
        is_best = score == score and score > best        
        if is_best:
            if self.select == "recall":
                self.best_recall = float(score)
            else:
                self.best_val = float(val_parts.get("total"))

        if ckpt is not None:
            torch.save(ckpt, os.path.join(self.out_dir, "last.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(self.out_dir, "best.pt"))
        return is_best, row

    _LOSS_W = 17        

    def _loss_keys(self, train_parts, val_parts):
        keys = [k for k in train_parts if k != "total"]
        for k in val_parts:
            if k != "total" and k not in keys:
                keys.append(k)
        return keys if len(keys) > 1 else []

    def header(self, loss_keys=None):
        if loss_keys is None:
            self._header_done = False
            return
        cols = [f"{'it':>6}", f"{'total tr/val':>{self._LOSS_W}}"]
        for k in loss_keys:
            cols.append(f"{k + ' tr/val':>{self._LOSS_W}}")
        cols.append(f"{'model R/allK':>13}")
        cols.append(f"{'sum R/allK':>13}")
        if self.train_sigma:
            cols += [f"{'σcorr':>7}", f"{'σ/err':>7}"]
        cols.append(f"{'lr':>9}")
        line = " | ".join(cols)
        print("\n" + line)
        print("-" * len(line))
        self._header_done = True

    def print_row(self, it, train_parts, val_parts, metrics, lr):
        keys = self._loss_keys(train_parts, val_parts)
        if not getattr(self, "_header_done", False):
            self.header(keys)

        def _pair(d1, d2, k):
            a, b = d1.get(k, float("nan")), d2.get(k, float("nan"))
            return f"{a:.4f}/{b:.4f}".rjust(self._LOSS_W)

        seg = [f"{it:>6}", _pair(train_parts, val_parts, "total")]
        for k in keys:                                   
            seg.append(_pair(train_parts, val_parts, k))
        seg.append(f"{metrics.get('recall', float('nan')):>6.3f}/"
                   f"{metrics.get('allk', float('nan')):.3f}")
        seg.append(f"{metrics.get('sum_recall', float('nan')):>6.3f}/"
                   f"{metrics.get('sum_allk', float('nan')):.3f}")
        if self.train_sigma:
            seg.append(f"{metrics.get('sigma_corr', float('nan')):>7.3f}")
            seg.append(f"{metrics.get('sigma_ratio', float('nan')):>7.3f}")
        seg.append(f"{lr:>9.2e}")
        print(" | ".join(seg), flush=True)

    def finish(self, per_snr, extra=None):
        crit = ("val_recall@K@%gdB" % self.monitor_snr_db) if self.select == "recall" \
            else "val_loss"
        metrics = {
            "stage": self.stage,
            "best_criterion": crit,
            "best_recall": (self.best_recall if np.isfinite(self.best_recall) else None),
            "best_val_loss": (self.best_val if np.isfinite(self.best_val) else None),
            "train_sigma": self.train_sigma,
            "monitor_snr_db": self.monitor_snr_db,
            "recall_L": self.recall_L,
            "per_snr": per_snr,                
            "train_log": self.train_log,
        }
        if extra:
            metrics.update(extra)
        with open(os.path.join(self.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str, ensure_ascii=False)

        if self.tb is not None:
            for snr, m in per_snr.items():
                for k, v in m.items():
                    try:
                        self.tb.add_scalar(f"final_{k}/vs_snr", float(v), int(float(snr)))
                    except (TypeError, ValueError):
                        continue
            self.tb.close()

        return metrics
