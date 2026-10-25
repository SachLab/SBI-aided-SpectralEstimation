import numpy as np
import torch
from torch.utils.data import Dataset

from config import resolve_probe_angles, signal_dim as _signal_dim
from dataset import rate_ref_hz, rate_summary
from synthesis_dataset import (
    amplitude_generation,
    sample_frequencies_hz,
    make_sampling_times,
    drifting_phase,
    draw_speed_fluct,
    draw_speed_ramp,
)


def n_windows(cfg):
    n_rev = int(cfg["data"]["n_rev"])
    n_rev_seq = int(cfg["crosswin"]["n_rev_seq"])
    hop = int(cfg["crosswin"]["hop_rev"])
    if n_rev_seq < n_rev:
        raise ValueError(f"n_rev_seq={n_rev_seq} < window n_rev={n_rev}")
    return (n_rev_seq - n_rev) // hop + 1


def _add_noise_norm(y, snr_db, rng, amps=None):
    if snr_db is not None:
        if amps is not None:
            noise_p = float(np.sum(np.asarray(amps) ** 2)) / (10.0 ** (snr_db / 10.0))
        else:
            noise_p = np.mean(np.abs(y) ** 2) / (10.0 ** (snr_db / 10.0))
        y = y + np.sqrt(noise_p / 2.0) * (
            rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    rms = np.sqrt(np.mean(np.abs(y) ** 2))
    return (y / (rms + 1e-12)).astype(np.complex64)


def _one_window_signal(freqs, amps, theta, t_n, snr_db, rng):
    phase = 2j * np.pi * freqs[:, None] * t_n[None, :]      # [K,N]
    camp = (amps * np.exp(1j * theta))[:, None]            # [K,1]
    y = (camp * np.exp(phase)).sum(axis=0)                 # [N]
    return _add_noise_norm(y, snr_db, rng, amps=amps)


class CrossWinDataset(Dataset):
    def __init__(self, cfg, size, seed=0, snr_db=None):
        self.cfg = cfg
        self.size = int(size)
        self.seed = int(seed)
        self.snr_db = snr_db
        self.probe_angles = resolve_probe_angles(cfg)
        self.n_probe = cfg["data"]["n_probes"]
        self.n_rev = int(cfg["data"]["n_rev"])      # window length        
        self.hop = int(cfg["crosswin"]["hop_rev"])          
        self.n_win = n_windows(cfg)
        self.n_rev_seq = int(cfg["crosswin"]["n_rev_seq"])  # the sequence length
        self.f_rot_hz = float(cfg["data"]["f_rot_hz"])          # fallback (f_rot_range=None 时)
        self.f_rot_range = cfg["data"].get("f_rot_range")       
        self.ref_hz = rate_ref_hz(cfg)     
        self.N = _signal_dim(cfg)                           #  n_probe*n_rev
        dcfg = cfg["data"]
        self.regime = dcfg["regime"]
        self.dcfg = dcfg
        self.freq_drift = bool(dcfg.get("freq_drift", False))
        self.drift_coupling = float(dcfg.get("drift_coupling", 0.5))
        self.drift_coupling_range = dcfg.get("drift_coupling_range")

    def __len__(self):
        return self.size

    def _generate(self, rng):
        scfg = self.cfg["signal"]
        dcfg = self.cfg["data"]
        K_MAX = int(scfg["num_components"])
        _kr = scfg.get("k_range")
        if _kr is not None:
            K = int(rng.integers(int(_kr[0]), min(int(_kr[1]), K_MAX) + 1))
        else:
            K = K_MAX
        freqs = sample_frequencies_hz(
            K, scfg["min_sep_hz"], scfg["f_lo"], scfg["f_hi"], scfg["distance"], rng)
        amps = amplitude_generation(K, scfg["amplitude"], scfg["floor_amplitude"], rng)
        _ff, _fa = scfg.get("fixed_freqs"), scfg.get("fixed_amps")
        if _ff is not None:
            freqs = np.asarray(_ff, dtype=float)
            assert freqs.size == K, f"fixed_freqs length {freqs.size} != num_components {K}"
        if _fa is not None:
            amps = np.asarray(_fa, dtype=float)
            assert amps.size == K, f"fixed_amps length {amps.size} != num_components {K}"
        theta = rng.random(K) * 2.0 * np.pi

        snr = self.snr_db
        if isinstance(snr, dict):
            snr = float(rng.uniform(float(snr["min"]), float(snr["max"])))
        elif isinstance(snr, (list, tuple)):
            snr = float(snr[int(rng.integers(len(snr)))]) if len(snr) else None

        n_probe = self.n_probe
        frr = self.f_rot_range
        f_rot = float(rng.uniform(frr[0], frr[1])) if frr else self.f_rot_hz
        N_seq = n_probe * self.n_rev_seq
        fluct_mode, speed_fluct = draw_speed_fluct(dcfg, rng)   
        speed_ramp = draw_speed_ramp(dcfg, f_rot, rng)   
        t_seq, rev_speeds_seq = make_sampling_times(
            N_seq, self.regime, self.probe_angles, f_rot,
            speed_fluct, rng, speed_ramp=speed_ramp,
            fluct_mode=fluct_mode)                     # [N_seq], [n_rev_seq]

        camp = (amps * np.exp(1j * theta))[:, None]
        if self.freq_drift:
            probe_phase = np.deg2rad(np.asarray(self.probe_angles)) / (2.0 * np.pi)
            ev = np.arange(N_seq)
            rev_cont = (ev // n_probe).astype(np.float64) + probe_phase[ev % n_probe]  # [N_seq]
            rev_mid = 0.5 * (rev_cont[0] + rev_cont[-1])
            cpr = self.drift_coupling_range
            coupling = float(rng.uniform(cpr[0], cpr[1])) if cpr else self.drift_coupling
            dpr = speed_ramp * coupling * (freqs / f_rot)                       # [K] Hz/rev
            phase_seq = drifting_phase(freqs, dpr, t_seq, rev_cont)             # [K,N_seq]
            y_seq = (camp * np.exp(1j * phase_seq)).sum(axis=0)                 
        else:
            phase_seq = 2.0 * np.pi * freqs[:, None] * t_seq[None, :]           
            y_seq = (camp * np.exp(1j * phase_seq)).sum(axis=0)                 
        y_seq_noisy = _add_noise_norm(y_seq, snr, rng, amps=amps)

        signals, t_ns, taus, rev_norms, f_wins, summaries = [], [], [], [], [], []
        for m in range(self.n_win):
            start_rev = m * self.hop                         
            r0, r1 = start_rev, start_rev + self.n_rev      
            p0, p1 = r0 * n_probe, r1 * n_probe              
            t_abs = t_seq[p0:p1]                             
            t_win = t_abs - t_abs[0]                         
            rev_win = rev_speeds_seq[r0:r1]                  
            if self.freq_drift:
                rev_c = 0.5 * (r0 + r1)
                f_win_m = freqs + dpr * (rev_c - rev_mid)    
            else:
                f_win_m = freqs                              
            y = y_seq_noisy[p0:p1]                           
            signals.append(np.stack([y.real, y.imag], axis=0).astype(np.float32))
            t_ns.append(t_win.astype(np.float32))
            rev_norms.append((rev_win / self.ref_hz - 1.0).astype(np.float32))   # FiLM per_rev
            summaries.append(rate_summary(rev_win, float(rev_win[0]),
                                          self.ref_hz).astype(np.float32))       # FiLM summary
            taus.append(float(start_rev + self.n_rev / 2.0)) 
            f_wins.append(np.sort(f_win_m).astype(np.float32))
        order = np.argsort(freqs)
        camp_sorted = (amps * np.exp(1j * theta))[order]     
        return {
            "freqs": np.sort(freqs).astype(np.float32),      
            "amp_true": np.abs(camp_sorted).astype(np.float32),   
            "camp_re": camp_sorted.real.astype(np.float32),  
            "camp_im": camp_sorted.imag.astype(np.float32),  
            "f_win": np.stack(f_wins, axis=0),               
            "signal": np.stack(signals, axis=0),             
            "t_n": np.stack(t_ns, axis=0),                   
            "rev_speeds_norm": np.stack(rev_norms, axis=0),  
            "rate_summary": np.stack(summaries, axis=0),     
            "tau": np.asarray(taus, dtype=np.float32),       
            "y_full": np.stack([y_seq_noisy.real, y_seq_noisy.imag], 0).astype(np.float32),  # [2,N_seq]
            "t_full": (t_seq - t_seq[0]).astype(np.float32), 
            "snr_db": np.float32(snr if snr is not None else np.nan),  
            "f_rot": np.float32(f_rot),  
            "K": K,
        }

    def _encode(self, s):
        M, K = self.n_win, s["K"]
        K_MAX = int(self.cfg["signal"]["num_components"])
        f_pad = np.full((M, K_MAX), -1e4, dtype=np.float32)
        f_pad[:, :K] = s["f_win"]                            

        def _pad1(v):
            out = np.zeros(K_MAX, dtype=np.float32)
            out[:K] = np.asarray(v, dtype=np.float32)
            return out

        return {
            "amp_true": torch.as_tensor(_pad1(s["amp_true"])),   
            "camp_re": torch.as_tensor(_pad1(s["camp_re"])),     
            "camp_im": torch.as_tensor(_pad1(s["camp_im"])),     
            "signal": torch.as_tensor(s["signal"]),          
            "t_n": torch.as_tensor(s["t_n"]),               
            "rev_speeds_norm": torch.as_tensor(s["rev_speeds_norm"]), 
            "rate_summary": torch.as_tensor(s["rate_summary"]),        
            "tau": torch.as_tensor(s["tau"]),                
            "f_win": torch.as_tensor(f_pad),                 
            "y_full": torch.as_tensor(s["y_full"]),          
            "t_full": torch.as_tensor(s["t_full"]),          
            "snr_db": torch.as_tensor(s["snr_db"]),         
            "f_rot": torch.as_tensor(s["f_rot"]),            
            "n_freq": torch.as_tensor(K, dtype=torch.long),
        }

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        return self._encode(self._generate(rng))
