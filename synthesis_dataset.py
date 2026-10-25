import numpy as np

def amplitude_generation(shape, amplitude="normal_floor", floor_amplitude=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if amplitude == "normal_floor":
        return np.abs(rng.standard_normal(shape)) + floor_amplitude
    if amplitude == "normal":
        return np.abs(rng.standard_normal(shape))
    if amplitude == "uniform":
        return rng.random(shape) * (1 - floor_amplitude) + floor_amplitude
    if amplitude == "equal":
        return np.ones(shape, dtype=float)
    raise ValueError(f"unknown amplitude={amplitude!r}")

def sample_frequencies_hz(nf, min_sep_hz, f_lo, f_hi, distance, rng, scale_frac=0.05):
    band = f_hi - f_lo

    if distance == "uniform":
        for _ in range(1000):
            cand = np.sort(f_lo + rng.random(nf) * band)
            if nf < 2 or np.min(np.diff(cand)) >= min_sep_hz:
                return cand
        raise RuntimeError(
            "could not sample separated frequencies; relax min_sep_hz / widen band"
        )

    if distance == "normal":
        for _ in range(500):
            first = f_lo + rng.random() * band
            vals = [first]
            ok = True
            for _i in range(1, nf):
                d = abs(rng.normal(scale=scale_frac * band)) + min_sep_hz
                nxt = vals[-1] + d
                if nxt >= f_hi:
                    ok = False
                    break
                vals.append(nxt)
            if ok and len(vals) == nf:
                return np.asarray(vals, dtype=np.float64)
        return np.linspace(
            f_lo, min(f_hi, f_lo + (nf - 1) * min_sep_hz + 1e-6), nf
        )

    if distance == "exact_sep":
        span = (nf - 1) * min_sep_hz
        if span >= band:
            raise ValueError(
                f"exact_sep: (nf-1)*min_sep_hz={span} is larger then band width {band}")
        first = f_lo + rng.random() * (band - span)
        return first + np.arange(nf, dtype=np.float64) * min_sep_hz

    raise ValueError(
        f"unknown distance={distance!r} (use 'normal' / 'uniform' / 'exact_sep')")


def draw_speed_knob(dcfg, name, rng):
    if not dcfg.get(f"{name}_sample", False):
        return float(dcfg.get(name, 0.0))
    rg = dcfg.get(f"{name}_range")
    if rg is None:
        raise ValueError(f"{name}_sample=True but {name}_range is missing")
    return float(rng.uniform(float(rg[0]), float(rg[1])))


def draw_speed_ramp(dcfg, f_rot_hz, rng):
    mode = dcfg.get("speed_ramp_mode", "rel")
    if mode == "rel":
        if dcfg.get("speed_ramp_sample", False):
            lo, hi = dcfg["speed_ramp_range"]
            return float(rng.uniform(float(lo), float(hi)))
        return float(dcfg.get("speed_ramp", 0.0))
    elif mode == "rpm":
        if dcfg.get("speed_ramp_rpm_sample", False):
            lo, hi = dcfg["speed_ramp_rpm_range"]
            rpm_per_rev = float(rng.uniform(float(lo), float(hi)))
        else:
            rpm_per_rev = float(dcfg["speed_ramp_rpm"])
        return rpm_per_rev / (float(f_rot_hz) * 60.0)
    raise ValueError(f"unknown speed_ramp_mode={mode!r} (should be 'rel' / 'rpm')")


def draw_speed_fluct(dcfg, rng):
    mode = dcfg.get("speed_fluct_mode", "rw")
    if mode == "rw":
        if dcfg.get("speed_fluct_sample", False):
            lo, hi = dcfg["speed_fluct_range"]
            return mode, float(rng.uniform(float(lo), float(hi)))
        return mode, float(dcfg.get("speed_fluct", 0.0))
    elif mode == "iid":
        if dcfg.get("speed_fluct_iid_sample", False):
            lo, hi = dcfg["speed_fluct_iid_range"]
            return mode, float(rng.uniform(float(lo), float(hi)))
        return mode, float(dcfg["speed_fluct_iid"])
    raise ValueError(f"unknown speed_fluct_mode={mode!r} (应为 'rw' / 'iid')")


FLUCT_KEYS = {"rw": "speed_fluct", "iid": "speed_fluct_iid"}
RAMP_KEYS = {"rel": "speed_ramp", "rpm": "speed_ramp_rpm"}


def fluct_mode_of(dcfg):
    mode = dcfg.get("speed_fluct_mode", "rw")
    if mode not in FLUCT_KEYS:
        raise ValueError(f"unknown speed_fluct_mode={mode!r} (shoul be 'rw' / 'iid')")
    return mode


def ramp_mode_of(dcfg):
    mode = dcfg.get("speed_ramp_mode", "rel")
    if mode not in RAMP_KEYS:
        raise ValueError(f"unknown speed_ramp_mode={mode!r} (should be 'rel' / 'rpm')")
    return mode


def apply_speed_mode_env(dcfg):
    import os
    fm = os.environ.get("FLUCT_MODE", "")
    rm = os.environ.get("RAMP_MODE", "")
    if fm:
        dcfg["speed_fluct_mode"] = fm
        fluct_mode_of(dcfg)
    if rm:
        dcfg["speed_ramp_mode"] = rm
        ramp_mode_of(dcfg)
    return dcfg


def active_speed_fluct(dcfg, default=0.0):
    return float(dcfg.get(FLUCT_KEYS[fluct_mode_of(dcfg)], default))


def active_speed_ramp(dcfg, default=0.0):
    return float(dcfg.get(RAMP_KEYS[ramp_mode_of(dcfg)], default))


def default_speed_fluct(dcfg, rw_default):
    return float(rw_default) if fluct_mode_of(dcfg) == "rw" else active_speed_fluct(dcfg)


def default_speed_ramp(dcfg, rel_default):
    return float(rel_default) if ramp_mode_of(dcfg) == "rel" else active_speed_ramp(dcfg)


def pin_speed_fluct(dcfg, sigma):
    key = FLUCT_KEYS[fluct_mode_of(dcfg)]
    dcfg[key] = float(sigma)
    dcfg[f"{key}_sample"] = False
    return dcfg


def pin_speed_ramp(dcfg, value):
    key = RAMP_KEYS[ramp_mode_of(dcfg)]
    dcfg[key] = float(value)
    dcfg[f"{key}_sample"] = False
    return dcfg


def fixed_speed_knobs(dcfg, f_rot_hz):
    ramp = active_speed_ramp(dcfg)
    if ramp_mode_of(dcfg) == "rpm":
        ramp = ramp / (float(f_rot_hz) * 60.0)
    return fluct_mode_of(dcfg), active_speed_fluct(dcfg), ramp


def speed_profile_tag(dcfg, fluct=None, ramp=None):
    fm, rm = fluct_mode_of(dcfg), ramp_mode_of(dcfg)
    fl = active_speed_fluct(dcfg) if fluct is None else float(fluct)
    rp = active_speed_ramp(dcfg) if ramp is None else float(ramp)
    fs = f"fluct={fl:g}" if fm == "rw" else f"fluct[{fm}]={fl:g}"
    rs = f"ramp={rp:g}" if rm == "rel" else f"ramp[{rm}]={rp:g}rpm/rev"
    return f"{fs} {rs}"


def speed_mode_record(dcfg):
    rec = {}
    if fluct_mode_of(dcfg) != "rw":
        rec["speed_fluct_mode"] = fluct_mode_of(dcfg)
    if ramp_mode_of(dcfg) != "rel":
        rec["speed_ramp_mode"] = ramp_mode_of(dcfg)
    return rec


def speed_mode_ref(dcfg, fallback=None):
    ref = {**(fallback or {}), **dcfg}
    ref["speed_fluct_mode"] = dcfg.get("speed_fluct_mode", "rw")
    ref["speed_ramp_mode"] = dcfg.get("speed_ramp_mode", "rel")
    return apply_speed_mode_env(ref)


def set_speed_profile(dcfg, ref, fluct, ramp):
    dcfg["speed_fluct_mode"] = fluct_mode_of(ref)
    dcfg["speed_ramp_mode"] = ramp_mode_of(ref)
    pin_speed_fluct(dcfg, fluct)
    pin_speed_ramp(dcfg, ramp)
    return dcfg


def speed_mode_note(dcfg):
    rec = speed_mode_record(dcfg)
    if not rec:
        return ""
    return f"[fluct_mode={fluct_mode_of(dcfg)} ramp_mode={ramp_mode_of(dcfg)}] "


def speed_mode_suffix(dcfg):
    return "".join(f"_{v}" for v in speed_mode_record(dcfg).values())


def make_sampling_times(signal_dim, regime, probe_angles_deg, f_rot_hz, speed_fluct,
                        rng, speed_ramp=0.0, fluct_mode="rw"):
    n_rev = signal_dim // len(probe_angles_deg)
    if regime == "uniform":
        return (np.arange(signal_dim, dtype=np.float64),
                np.full(n_rev, f_rot_hz, dtype=np.float64))

    angles = np.asarray(probe_angles_deg, dtype=np.float64)
    n_probe = len(angles)
    probe_phase = np.deg2rad(angles) / (2.0 * np.pi)      
    event = np.arange(signal_dim)
    probe_idx = event % n_probe
    rev_idx = event // n_probe

    if regime == "probe":
        rev_speeds = np.full(n_rev, f_rot_hz, dtype=np.float64)
        t_rev = 1.0 / f_rot_hz
        t_n = (rev_idx.astype(np.float64) + probe_phase[probe_idx]) * t_rev
    elif regime == "offset":
        f_seq = f_rot_hz * (1.0 + speed_fluct * (2.0 * rng.random() - 1.0))
        rev_speeds = np.full(n_rev, f_seq, dtype=np.float64)
        t_rev = 1.0 / f_seq
        t_n = (rev_idx.astype(np.float64) + probe_phase[probe_idx]) * t_rev
    elif regime == "fluct":
        r = np.arange(n_rev, dtype=np.float64)
        if speed_fluct <= 0:
            rw = 0.0
        elif fluct_mode == "rw":
            rw = np.cumsum(speed_fluct * rng.standard_normal(n_rev))    
        elif fluct_mode == "iid":
            rw = speed_fluct * rng.standard_normal(n_rev)               
        else:
            raise ValueError(f"unknown fluct_mode={fluct_mode!r} (应为 'rw' / 'iid')")
        factor = 1.0 + speed_ramp * r + rw                        # [n_rev]
        rev_speeds = (f_rot_hz * factor).clip(min=1e-3)           # Hz
        periods = 1.0 / rev_speeds                                # [n_rev] 
        rev_start = np.concatenate([[0.0], np.cumsum(periods)[:-1]])
        t_n = rev_start[rev_idx] + probe_phase[probe_idx] * periods[rev_idx]
    else:
        raise ValueError(f"unknown regime={regime!r}")

    return t_n - t_n[0], rev_speeds

def drifting_phase(freqs_bar, dpr_hz_per_rev, t_n, rev_cont):
    rev_mid = 0.5 * (rev_cont[0] + rev_cont[-1])
    f_inst = freqs_bar[:, None] + dpr_hz_per_rev[:, None] * (rev_cont[None, :] - rev_mid)  
    dt = np.diff(t_n)                                               
    favg = 0.5 * (f_inst[:, 1:] + f_inst[:, :-1])                  
    dphi = 2.0 * np.pi * favg * dt[None, :]                        
    phase = np.concatenate([np.zeros((f_inst.shape[0], 1)), np.cumsum(dphi, axis=1)], axis=1)
    return phase                                                   


def generate_one_sample(
    signal_dim,
    num_components,
    min_sep_hz,
    f_lo,
    f_hi,
    distance,
    amplitude,
    floor_amplitude,
    regime,
    probe_angles_deg,
    f_rot_hz,
    speed_fluct,
    rng,
    snr_db=None,
    k_range=None,
    speed_ramp=0.0,
    freq_drift=False,
    drift_coupling=0.5,
    fluct_mode="rw",
):

    if k_range is not None:
        lo, hi = int(k_range[0]), int(k_range[1])
        hi = min(hi, int(num_components))           
        n_freq = int(rng.integers(lo, hi + 1))      
    else:
        n_freq = num_components                  

    freqs_active = sample_frequencies_hz(
        n_freq, min_sep_hz, f_lo, f_hi, distance, rng
    )
    amps = amplitude_generation(n_freq, amplitude, floor_amplitude, rng)
    theta = rng.random(n_freq) * 2.0 * np.pi

    t_n, rev_speeds = make_sampling_times(
        signal_dim, regime, probe_angles_deg, f_rot_hz, speed_fluct, rng,
        speed_ramp=speed_ramp, fluct_mode=fluct_mode
    )

    if not freq_drift:
        phase = 2.0 * np.pi * freqs_active[:, None] * t_n[None, :]    
    else:
        n_probe = len(probe_angles_deg)
        probe_phase = np.deg2rad(np.asarray(probe_angles_deg)) / (2.0 * np.pi)  # [n_probe]
        ev = np.arange(signal_dim)
        rev_cont = (ev // n_probe).astype(np.float64) + probe_phase[ev % n_probe]  # [N]
        dpr = speed_ramp * drift_coupling * (freqs_active / f_rot_hz)  # [K]
        phase = drifting_phase(freqs_active, dpr, t_n, rev_cont)       
    complex_amp = (amps * np.exp(1j * theta))[:, None]            # [K,1]
    y_clean = (complex_amp * np.exp(1j * phase)).sum(axis=0)     # [N] complex

    noise_power = 0.0
    if snr_db is not None:
        noise_power = float(np.sum(amps ** 2)) / (10.0 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(signal_dim) + 1j * rng.standard_normal(signal_dim)
        )
        y = y_clean + noise
    else:
        y = y_clean

    rms = np.sqrt(np.mean(np.abs(y) ** 2))
    y = y / (rms + 1e-12)

    freqs_padded = np.full(num_components, -1e4, dtype=np.float64)
    freqs_padded[:n_freq] = freqs_active
    amps_padded = np.zeros(num_components, dtype=np.float64)
    amps_padded[:n_freq] = amps

    return {
        "y": y.astype(np.complex64),
        "t_n": t_n.astype(np.float32),
        "rev_speeds": rev_speeds.astype(np.float32),
        "freqs_hz": freqs_padded.astype(np.float32),
        "amps": amps_padded.astype(np.float32),
        "n_freq": n_freq,
    }
