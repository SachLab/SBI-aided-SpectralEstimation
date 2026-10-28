import math
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

TWO_PI = 2.0 * math.pi

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

for _v in _BLAS_VARS:
    os.environ.setdefault(_v, "1")

_DEFAULT_JOBS = min(16, os.cpu_count() or 1)


def _init_worker():
    for v in _BLAS_VARS:
        os.environ[v] = "1"


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_tensor(x, like):
    if not hasattr(like, "detach"):
        return x
    import torch
    if np.issubdtype(np.asarray(x).dtype, np.integer):
        return torch.as_tensor(x, device=like.device, dtype=torch.long)
    if np.iscomplexobj(x):
        return torch.as_tensor(x, device=like.device, dtype=torch.complex64)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _mean_atoms(mu, sigma2, t):
    return (np.exp(1j * TWO_PI * np.outer(t, mu))
            * np.exp(-2.0 * math.pi**2 * np.outer(t**2, sigma2)))


def _expected_stats(mu, sigma2, t, y):
    Abar = _mean_atoms(mu, sigma2, t)
    J = Abar.conj().T @ Abar
    np.fill_diagonal(J, float(len(t)))
    h = Abar.conj().T @ y
    return Abar, J, h


def _refresh_column(Abar, J, h, k, mu, sigma2, t, y):
    ak = (np.exp(1j * TWO_PI * t * mu[k])
          * np.exp(-2.0 * math.pi**2 * t**2 * sigma2[k]))
    Abar[:, k] = ak
    col = Abar.conj().T @ ak            
    J[:, k] = col
    J[k, :] = col.conj()
    J[k, k] = float(len(t))             
    h[k] = np.vdot(ak, y)
    return Abar, J, h


def _amplitude_posterior(J, h, tau2, sw2, S):
    idx = np.asarray(S, dtype=int)
    A = J[np.ix_(idx, idx)] / sw2 + np.diag(1.0 / tau2[idx])
    C = np.linalg.inv(A)
    c = C @ h[idx] / sw2
    return A, c, C


def _subset_score(J, h, tau2, sw2, rho, S):
    idx = np.asarray(S, dtype=int)
    A = J[np.ix_(idx, idx)] / sw2 + np.diag(1.0 / tau2[idx])
    L = np.linalg.cholesky(A)
    z = np.linalg.solve(L, h[idx])
    logit = np.log(rho[idx] / (1.0 - rho[idx]))
    return (np.real(z.conj() @ z) / sw2**2
            - 2.0 * np.log(np.real(np.diag(L))).sum()
            - np.log(tau2[idx]).sum()
            + logit.sum())


def _subset_posterior(J, h, tau2, sw2, rho, subsets):
    F = np.array([_subset_score(J, h, tau2, sw2, rho, S) for S in subsets])
    m = F.max()
    w = np.exp(F - m)
    qS = w / w.sum()
    return F, qS, m + np.log(w.sum())


def _residuals(Abar, y, subsets, qS, chats, covs, Kc, sub_idx=None):
    idx = np.asarray(subsets, dtype=int) if sub_idx is None else sub_idx
    w = np.asarray(qS, dtype=float)                     # [B]
    c = np.asarray(chats)                               # [B,K]
    C = np.asarray(covs)                                # [B,K,K]

    # alpha[kappa] = sum_{S ni kappa} q(S) conj(c_kappa^S)
    alpha = np.zeros(Kc, dtype=np.complex128)
    np.add.at(alpha, idx.ravel(), (w[:, None] * np.conj(c)).ravel())

    # V[ell,kappa] = sum_S q(S) (c_ell^S conj(c_kappa^S) + C_{ell,kappa}^S),
    # with the ell == kappa self term excluded.
    blk = c[:, :, None] * np.conj(c)[:, None, :] + C    # [B,K,K], (ell, kappa)
    K = idx.shape[1]
    eye = np.eye(K, dtype=bool)
    blk[:, eye] = 0.0                                   # drop the self term
    blk *= w[:, None, None]
    rows = np.broadcast_to(idx[:, :, None], blk.shape)
    cols = np.broadcast_to(idx[:, None, :], blk.shape)
    V = np.zeros((Kc, Kc), dtype=np.complex128)
    np.add.at(V, (rows.ravel(), cols.ravel()), blk.ravel())

    return y[:, None] * alpha[None, :] - Abar @ V


def _g_value(mu_k, sigma2_k, r, t, sw2):
    damp = np.exp(-2.0 * math.pi**2 * sigma2_k * t**2)
    return (2.0 * np.sum(damp * np.real(r * np.exp(-1j * TWO_PI * mu_k * t))) / sw2
            + 0.5 * np.log(sigma2_k))


def _g_grad(mu_k, sigma2_k, r, t, sw2):
    beta = 2.0 * np.abs(r) * np.exp(-2.0 * math.pi**2 * sigma2_k * t**2) / sw2
    theta = np.angle(r) - TWO_PI * mu_k * t
    c = np.cos(theta)
    s = np.sin(theta)
    d_mu = np.sum(beta * TWO_PI * t * s)
    d_s2 = -np.sum(beta * 2.0 * math.pi**2 * t**2 * c) + 0.5 / sigma2_k
    return np.array([d_mu, d_s2])


def _update_frequency_point(mu_k, r, t, sw2, band=None):
    def neg(x):
        mu = x[0]
        th = TWO_PI * mu * t
        val = 2.0 * np.sum(np.real(r * np.exp(-1j * th))) / sw2
        beta = 2.0 * np.abs(r) / sw2
        theta = np.angle(r) - th
        d_mu = np.sum(beta * TWO_PI * t * np.sin(theta))
        return -val, -np.array([d_mu])

    lo_mu, hi_mu = (None, None) if band is None else band
    old = -neg(np.array([mu_k]))[0]
    res = minimize(neg, np.array([float(mu_k)]), jac=True,
                   method="L-BFGS-B", bounds=[(lo_mu, hi_mu)])
    if res.success and -res.fun >= old:
        return float(res.x[0]), 0.0
    return float(mu_k), 0.0


def _update_frequency(mu_k, sigma2_k, r, t, sw2, band=None):
    lo_mu, hi_mu = (None, None) if band is None else band
    tm = float(np.max(np.abs(t)))
    s2_underflow = 745.0 / (2.0 * math.pi ** 2 * max(tm, 1e-12) ** 2)
    hi_s2 = None
    if lo_mu is not None and hi_mu is not None:
        hw = 0.5 * (float(hi_mu) - float(lo_mu))
        if hw > 0:
            hi_s2 = np.log(min(hw ** 2, s2_underflow))
    lo_s2 = np.log(1e-12)
    if hi_s2 is None:                  
        hi_s2 = np.log(min(1e6, s2_underflow))
    bounds = [(lo_mu, hi_mu), (lo_s2, hi_s2)]

    def neg(x):
        s2 = np.exp(x[1])
        val = _g_value(x[0], s2, r, t, sw2)
        grad = _g_grad(x[0], s2, r, t, sw2)
        return -val, -np.array([grad[0], grad[1] * s2])

    old = _g_value(mu_k, sigma2_k, r, t, sw2)
    res = minimize(neg, np.array([mu_k, np.log(sigma2_k)]), jac=True,
                   method="L-BFGS-B", bounds=bounds)
    if res.success and -res.fun >= old:
        return float(res.x[0]), float(np.exp(res.x[1]))
    return float(mu_k), float(sigma2_k)


def _model_avg_amplitude(subsets, qS, chats, covs, Kc, eps,
                         S_star=None, c_star=None, C_star=None):
    c_map = np.zeros(Kc, dtype=np.complex128)
    v_map = np.zeros(Kc, dtype=float)
    if not AMP_MODEL_AVG:
        c_map[S_star] = c_star
        v_map[S_star] = np.real(np.diag(C_star))
        return c_map, v_map
    for kk in range(Kc):
        wsum = 0.0
        m1 = 0.0 + 0.0j
        m2 = 0.0                     # sum_S q(S) (C_kk^(S) + |c_k^(S)|^2)
        for S, w, c, C in zip(subsets, qS, chats, covs):
            if kk not in S:
                continue
            j = S.index(kk)
            wsum += w
            m1 += w * c[j]
            m2 += w * (np.real(C[j, j]) + abs(c[j]) ** 2)
        if wsum > eps:
            c_map[kk] = m1 / wsum
            v_map[kk] = max(m2 / wsum - abs(c_map[kk]) ** 2, 0.0)
    return c_map, v_map


def refine_stage2_tex_instance(y, t, rho, mu0, sigma0, K, n_sweeps=40,
                               tol=1e-8, bands=None, verbose=False,
                               eps=1e-12, jacobi=None, trace=False):
    if jacobi is None:
        jacobi = JACOBI
    y = np.asarray(y, dtype=np.complex128)
    t = np.asarray(t, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if RHO_UNIFORM:                     
        rho = np.full_like(rho, 0.5)
    if not NOCLIP:
        rho = rho.clip(1e-6, 1.0 - 1e-6)
    mu = np.asarray(mu0, dtype=float).copy()
    sigma2 = np.asarray(sigma0, dtype=float) ** 2
    if not NOCLIP:
        sigma2 = np.maximum(sigma2, eps)
    if FREQ_SIGMA0:                     
        sigma2 = np.zeros_like(sigma2)
    Kc = len(mu)
    N = len(t)
    subsets = [list(S) for S in combinations(range(Kc), K)]
    sub_idx = np.asarray(subsets, dtype=int)        # [B,K] view, reused every sweep

    Phi0 = np.exp(1j * TWO_PI * np.outer(t, mu))
    c0 = np.linalg.lstsq(Phi0, y, rcond=None)[0]
    tau2 = np.abs(c0) ** 2
    sw2 = float(np.mean(np.abs(y - Phi0 @ c0) ** 2))
    if not NOCLIP:
        tau2 = np.maximum(tau2, eps)
        sw2 = max(sw2, eps)

    hist = []
    mu_hist, amp_hist = [], []
    for sweep in range(n_sweeps):
        Abar, J, h = _expected_stats(mu, sigma2, t, y)
        if jacobi:
            chats, covs = [], []
            for S in subsets:
                _, c, C = _amplitude_posterior(J, h, tau2, sw2, S)
                chats.append(c)
                covs.append(C)
            F, qS, _ = _subset_posterior(J, h, tau2, sw2, rho, subsets)
            R = _residuals(Abar, y, subsets, qS, chats, covs, Kc, sub_idx)
            for k in range(Kc):
                band = None if bands is None else bands[k]
                if FREQ_SIGMA0:
                    mu[k], sigma2[k] = _update_frequency_point(
                        mu[k], R[:, k], t, sw2, band)
                else:
                    mu[k], sigma2[k] = _update_frequency(mu[k], sigma2[k], R[:, k],
                                                         t, sw2, band)
        else:
            for k in range(Kc):
                if k:                   # k=0 uses the stats just built above
                    Abar, J, h = _refresh_column(Abar, J, h, k - 1, mu, sigma2, t, y)
                chats, covs = [], []
                for S in subsets:
                    _, c, C = _amplitude_posterior(J, h, tau2, sw2, S)
                    chats.append(c)
                    covs.append(C)
                F, qS, _ = _subset_posterior(J, h, tau2, sw2, rho, subsets)
                R = _residuals(Abar, y, subsets, qS, chats, covs, Kc, sub_idx)
                band = None if bands is None else bands[k]
                if FREQ_SIGMA0:
                    mu[k], sigma2[k] = _update_frequency_point(
                        mu[k], R[:, k], t, sw2, band)
                else:
                    mu[k], sigma2[k] = _update_frequency(mu[k], sigma2[k], R[:, k],
                                                         t, sw2, band)

        Abar, J, h = _expected_stats(mu, sigma2, t, y)
        chats, covs = [], []
        for S in subsets:
            _, c, C = _amplitude_posterior(J, h, tau2, sw2, S)
            chats.append(c)
            covs.append(C)
        F, qS, lse = _subset_posterior(J, h, tau2, sw2, rho, subsets)

        pi = np.array([sum(w for S, w in zip(subsets, qS) if k in S) for k in range(Kc)])
        for k in range(Kc):
            if pi[k] <= eps:
                continue
            _t2 = sum(
                w * (abs(c[S.index(k)]) ** 2 + np.real(C[S.index(k), S.index(k)]))
                for S, w, c, C in zip(subsets, qS, chats, covs) if k in S
            ) / pi[k]
            tau2[k] = _t2 if NOCLIP else max(_t2, eps)

        yn2 = float(np.real(y.conj() @ y))
        _sw2 = sum(
            w * (yn2
                 - 2.0 * np.real(c.conj() @ h[np.asarray(S)])
                 + np.real(np.trace((np.outer(c, c.conj()) + C)
                                    @ J[np.ix_(np.asarray(S), np.asarray(S))])))
            for S, w, c, C in zip(subsets, qS, chats, covs)
        ) / N
        sw2 = _sw2 if NOCLIP else max(_sw2, eps)

        _, J, h = _expected_stats(mu, sigma2, t, y)
        F, qS, lse = _subset_posterior(J, h, tau2, sw2, rho, subsets)
        L = lse if FREQ_SIGMA0 else (lse + 0.5 * np.log(sigma2).sum())
        hist.append(float(L))
        if trace:
            _js = int(np.argmax(F))
            _c, _ = _model_avg_amplitude(subsets, qS, chats, covs, Kc, eps,
                                         subsets[_js], chats[_js], covs[_js])
            mu_hist.append(mu.copy())
            amp_hist.append(np.abs(_c))
        if verbose:
            print(f"  sweep{sweep:>3d} elbo={hist[-1]: .6f} sw2={sw2:.3e}")
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * max(1.0, abs(hist[-1])):
            break

    Abar, J, h = _expected_stats(mu, sigma2, t, y)
    F, qS, _ = _subset_posterior(J, h, tau2, sw2, rho, subsets)
    pi = np.array([sum(w for S, w in zip(subsets, qS) if k in S) for k in range(Kc)])
    S_star = subsets[int(np.argmax(F))]
    _, c_star, C_star = _amplitude_posterior(J, h, tau2, sw2, S_star)
    c_map, v_map = _model_avg_amplitude(subsets, qS, chats, covs, Kc, eps,
                                        S_star, c_star, C_star)
    out_trace = {
        "mu": mu,
        "sigma": np.sqrt(sigma2),
        "tau2": tau2,
        "sigma_n2": sw2,
        "subsets": np.asarray(subsets, dtype=int),
        "qS": qS,
        "subset_scores": F,
        "pi": pi,
        "S_star": np.asarray(S_star, dtype=int),
        "amp_c": c_map,
        "amp": np.abs(c_map),
        "amp_var": v_map,
        "amp_sigma": np.sqrt(np.maximum(v_map, 0.0)),
        "score": pi * np.abs(c_map),
        "elbo": np.asarray(hist),
    }
    if trace:
        out_trace["mu_hist"] = np.asarray(mu_hist)
        out_trace["amp_hist"] = np.asarray(amp_hist)
    return out_trace


def _refine_one(args):
    """Top-level worker (must be importable for multiprocessing pickling)."""
    yb, tb, rb, mb, sb, K, n_sweeps, tol, b_bands, verbose, jacobi, trace = args
    return refine_stage2_tex_instance(yb, tb, rb, mb, sb, K, n_sweeps=n_sweeps,
                                      tol=tol, bands=b_bands, verbose=verbose,
                                      jacobi=jacobi, trace=trace)


BAND_FRAC = float(os.environ.get("TEX_BAND_FRAC", 0.5))
BAND_GRID_HZ_DEFAULT = float(os.environ.get("TEX_BAND_GRID_HZ", 2.0))
JACOBI = os.environ.get("TEX_JACOBI", "1").lower() in ("1", "true", "yes")
NOCLIP = os.environ.get("TEX_NOCLIP", "1").lower() in ("1", "true", "yes")
FREQ_SIGMA0 = os.environ.get("TEX_FREQ_SIGMA0", "0").lower() in ("1", "true", "yes")
RHO_UNIFORM = os.environ.get("TEX_RHO_UNIFORM", "0").lower() in ("1", "true", "yes")
AMP_MODEL_AVG = os.environ.get("TEX_AMP_MODEL_AVG", "1").lower() in ("1", "true", "yes")


def refine_stage2_tex_batch(init_mu, init_logs, rho_prior, y_full, t_full, K,
                            n_sweeps=40, tol=1e-8, bands=None,
                            verbose=False, n_jobs=None, band_hz=None,
                            jacobi=None, grid_step_hz=None, trace=False):
    like = init_mu
    mu0 = _as_numpy(init_mu)
    sigma0 = np.exp(_as_numpy(init_logs))
    rho = _as_numpy(rho_prior)
    y = _as_numpy(y_full)
    t = _as_numpy(t_full)
    B = mu0.shape[0]
    if band_hz is not None:
        hw = float(band_hz)
    else:
        step = (BAND_GRID_HZ_DEFAULT if grid_step_hz is None else float(grid_step_hz))
        hw = BAND_FRAC * step
    if bands is not None:
        band_np = _as_numpy(bands)
    elif hw > 0:
        band_np = np.stack([mu0 - hw, mu0 + hw], axis=-1)
    else:
        band_np = None

    if n_jobs is None:
        n_jobs = int(os.environ.get("TEX_JOBS", 0)) or _DEFAULT_JOBS
    n_jobs = max(1, min(int(n_jobs), B))

    jac = JACOBI if jacobi is None else bool(jacobi)
    tasks = [(y[b], t[b], rho[b], mu0[b], sigma0[b], K, n_sweeps, tol,
              None if band_np is None else band_np[b], verbose, jac, trace)
             for b in range(B)]

    if n_jobs == 1:
        outs = [_refine_one(a) for a in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs,
                                 initializer=_init_worker) as ex:
            outs = list(ex.map(_refine_one, tasks, chunksize=1))

    keys = ["mu", "sigma", "tau2", "pi", "amp", "amp_c", "amp_var",
            "amp_sigma", "score", "S_star"]
    out = {k: _maybe_tensor(np.stack([o[k] for o in outs], axis=0), like) for k in keys}
    out["sigma_n2"] = _maybe_tensor(np.asarray([o["sigma_n2"] for o in outs]), like)
    out["qS"] = _maybe_tensor(np.stack([o["qS"] for o in outs], axis=0), like)
    out["subset_scores"] = _maybe_tensor(np.stack([o["subset_scores"] for o in outs], axis=0), like)
    out["subsets"] = _maybe_tensor(outs[0]["subsets"], like)
    out["elbo"] = [o["elbo"] for o in outs]
    if trace:
        out["mu_hist"] = [o["mu_hist"] for o in outs]
        out["amp_hist"] = [o["amp_hist"] for o in outs]
    return out
