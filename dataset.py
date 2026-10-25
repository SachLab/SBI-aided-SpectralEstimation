import numpy as np
import torch
from torch.utils.data import Dataset

from config import grid_size, resolve_probe_angles, signal_dim as _signal_dim
from synthesis_dataset import draw_speed_fluct, draw_speed_ramp, generate_one_sample


def make_freq_grid(cfg):
    fcfg = cfg["frequency"]
    return np.linspace(fcfg["f_min"], fcfg["f_max"], grid_size(cfg)).astype(np.float32)


RATE_REF_HZ = 35.0  


def rate_ref_hz(cfg):
    return float(cfg["data"].get("rate_ref_hz", RATE_REF_HZ))


def rate_summary(rev_speeds, f_rot_hz, ref_hz=RATE_REF_HZ):
    s = rev_speeds.astype(np.float64) / f_rot_hz - 1.0
    n = len(s)
    idx = np.arange(n, dtype=np.float64)
    ic = idx - idx.mean()
    mean = s.mean()
    denom = (ic ** 2).sum()
    slope = (ic * (s - mean)).sum() / denom if denom > 0 else 0.0
    resid = s - (mean + slope * ic)
    abs_mean = rev_speeds.astype(np.float64).mean() / float(ref_hz) - 1.0
    return np.array([mean, slope, resid.var(), abs_mean], dtype=np.float32)


def film_input(cfg):
    return cfg["encoder"].get("film_input", "summary")


def film_input_dim(cfg):
    mode = film_input(cfg)
    if mode == "summary":
        return 4
    if mode == "per_rev":
        return int(cfg["data"]["n_rev"])
    raise ValueError(f"unknown encoder.film_input={mode!r} (Need summary / per_rev)")


class UnifiedBTTDataset(Dataset):
    def __init__(self, cfg, size, seed=0, snr_db=None):
        self.cfg = cfg
        self.ref_hz = rate_ref_hz(cfg)      
        self.size = int(size)
        self.seed = int(seed)
        self.snr_db = snr_db
        self.probe_angles = resolve_probe_angles(cfg)
        self.grid = make_freq_grid(cfg)

    def __len__(self):
        return self.size

    def _generate(self, rng):
        dcfg = self.cfg["data"]
        scfg = self.cfg["signal"]
        snr = self.snr_db
        if isinstance(snr, dict):
            lf = snr.get("low_frac", 0.0)
            if lf > 0 and rng.random() < lf:         
                snr = float(rng.uniform(snr.get("low_lo", snr["min"]),
                                        snr.get("low_hi", snr["min"] + 10.0)))
            else:                                     
                snr = float(rng.uniform(snr["min"], snr["max"]))
        elif isinstance(snr, (list, tuple)):
            snr = float(snr[int(rng.integers(len(snr)))]) if len(snr) else None
        frr = dcfg.get("f_rot_range")
        f_rot = float(rng.uniform(frr[0], frr[1])) if frr else float(dcfg["f_rot_hz"])
        cpr = dcfg.get("drift_coupling_range")
        coupling = float(rng.uniform(cpr[0], cpr[1])) if cpr else dcfg.get("drift_coupling", 0.5)
        fmode, sfluct = draw_speed_fluct(dcfg, rng)    
        sramp = draw_speed_ramp(dcfg, f_rot, rng)    
        sample = generate_one_sample(
            signal_dim=_signal_dim(self.cfg),
            num_components=scfg["num_components"],
            min_sep_hz=scfg["min_sep_hz"],
            f_lo=scfg["f_lo"],
            f_hi=scfg["f_hi"],
            distance=scfg["distance"],
            amplitude=scfg["amplitude"],
            floor_amplitude=scfg["floor_amplitude"],
            k_range=scfg.get("k_range"),        
            regime=dcfg["regime"],
            probe_angles_deg=self.probe_angles,
            f_rot_hz=f_rot,
            speed_fluct=sfluct,
            fluct_mode=fmode,
            rng=rng,
            snr_db=snr,
            speed_ramp=sramp,
            freq_drift=dcfg.get("freq_drift", False),
            drift_coupling=coupling,
        )
        sample["f_rot"] = f_rot          
        return sample

    def _encode(self, sample):
        y = sample["y"]
        rev_speeds = sample["rev_speeds"]
        signal = np.stack([y.real, y.imag], axis=0).astype(np.float32)
        f_rot = sample.get("f_rot", self.cfg["data"]["f_rot_hz"])
        summary = rate_summary(rev_speeds, f_rot, self.ref_hz)
        rev_norm = rev_speeds.astype(np.float32) / self.ref_hz - 1.0
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "t_n": torch.as_tensor(sample["t_n"], dtype=torch.float32),
            "rate_summary": torch.as_tensor(summary, dtype=torch.float32),        # [4]
            "rev_speeds_norm": torch.as_tensor(rev_norm, dtype=torch.float32),    # [n_rev]
            "freqs_hz": torch.as_tensor(sample["freqs_hz"], dtype=torch.float32),
            "amps": torch.as_tensor(sample["amps"], dtype=torch.float32),         
            "n_freq": torch.as_tensor(sample["n_freq"], dtype=torch.long),
        }

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        return self._encode(self._generate(rng))
