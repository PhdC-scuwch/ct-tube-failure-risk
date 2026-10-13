"""Label-free, stage-aligned degradation trend of the filament health index and the pairwise trend similarity
used by DTG-v2 (Sec. IV-D of the paper).

Each client turns its HI series into per-step degradation-rate terms (HI decrease per scan, causal moving average)
and bins them by HI level (the degradation stage) into `fine_bins` bins. Only the per-bin sums and counts leave the
client. The server compares two clients ONLY over the stage range both have covered (between the 1 % and 99 %
quantiles of each client's counts), at a resolution of B equal groups inside that range, with the Pearson
correlation of the mean rates. No lifecycle length, label or interpolation between trajectories is needed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def masked_pearson(a, b, ma, mb, min_overlap: int = 5) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.asarray(ma, bool) & np.asarray(mb, bool) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_overlap:
        return 0.0
    x = np.asarray(a, float)[m]
    y = np.asarray(b, float)[m]
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 1e-12 else 0.0


def weights_from_rho(rho: np.ndarray, include_self: bool = True) -> np.ndarray:
    sim = np.clip((rho + 1.0) / 2.0, 0.0, 1.0)
    K = sim.shape[0]
    if include_self:
        s = sim.mean(1)
    else:
        sim = sim.copy()
        np.fill_diagonal(sim, 0.0)
        s = sim.sum(1) / max(K - 1, 1)
    if s.sum() <= 0:
        return np.ones(K) / K
    return s / s.sum()


def _trailing_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.astype(float)
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def hi_rate_terms(hi: np.ndarray, t: np.ndarray, smooth: Optional[int] = None,
                  lag: Optional[int] = None, r_clip: Optional[float] = None, return_t: bool = False):
    """Per-step degradation-rate terms (HI units per time unit) and the HI level they belong to.
    Uses a causal trailing moving average, so a prefix of the trajectory gives the same terms."""
    hi, t = np.asarray(hi, float), np.asarray(t, float)
    fin = np.isfinite(hi)                 # HI is undefined before the filament estimator warms up
    hi, t = hi[fin], t[fin]
    T = len(hi)
    smooth = max(1, T // 100) if smooth is None else smooth
    lag = max(1, T // 200) if lag is None else lag
    hs = _trailing_ma(np.asarray(hi, float), smooth)
    if T <= lag:
        return (np.zeros(0),) * (3 if return_t else 2)
    a, b = hs[:-lag], hs[lag:]
    dt = (np.asarray(t, float)[lag:] - np.asarray(t, float)[:-lag])
    ok = np.isfinite(a) & np.isfinite(b) & (dt > 0)
    rate = (a[ok] - b[ok]) / dt[ok]
    if r_clip is not None:
        rate = np.clip(rate, -r_clip, r_clip)
    if return_t:
        return rate, a[ok], np.asarray(t, float)[:-lag][ok]
    return rate, a[ok]


def stage_sums(values: np.ndarray, coord: np.ndarray, bins: int = 100):
    """Fine-grained sufficient statistics of a profile `values` vs a stage coordinate in [0, 1]
    (HI level for rate profiles, true normalized RUL for bias profiles). Only these per-bin sums
    and counts leave a client (and they can be privatized, see dp.privatize_trend)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    k = np.clip(np.digitize(np.asarray(coord, float), edges) - 1, 0, bins - 1)
    return (np.bincount(k, weights=np.asarray(values, float), minlength=bins),
            np.bincount(k, minlength=bins).astype(float))


def covered_range(c: np.ndarray, q: float = 0.01):
    """Stage range a client covers: fine bins between the q and 1-q quantiles of its counts
    (robust to sparsely filled fine bins; works on noisy DP counts as well)."""
    tot = float(np.sum(c))
    if tot <= 0:
        return None
    cs = np.cumsum(c) / tot
    return int(np.searchsorted(cs, q, side="left")), int(np.searchsorted(cs, 1.0 - q, side="left"))


def pairwise_stage_rho(sk, ck, sj, cj, B: int = 20, min_count: float = 3, min_groups: int = 5,
                       q: float = 0.01) -> float:
    """Compare two profiles ONLY over the stage range both clients cover, at a fixed resolution of
    B groups inside that range (instead of fixed bins over [0, 1], which leaves a partially
    observed tube with only a few usable points). Groups with < min_count samples are masked."""
    rk, rj = covered_range(ck, q), covered_range(cj, q)
    if rk is None or rj is None:
        return 0.0
    lo, hi = max(rk[0], rj[0]), min(rk[1], rj[1])
    if hi - lo + 1 < min_groups:
        return 0.0
    a, b, m = [], [], []
    for g in np.array_split(np.arange(lo, hi + 1), min(B, hi - lo + 1)):
        nk, nj = ck[g].sum(), cj[g].sum()
        a.append(sk[g].sum() / max(nk, 1e-12))
        b.append(sj[g].sum() / max(nj, 1e-12))
        m.append(nk >= min_count and nj >= min_count)
    m = np.array(m)
    return masked_pearson(np.array(a), np.array(b), m, m, min_groups)


def dtg_weights_pairwise(Ss, Cs, B: int = 20, include_self: bool = True, min_count: float = 3,
                         min_groups: int = 5):
    K = len(Ss)
    rho = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            rho[i, j] = rho[j, i] = pairwise_stage_rho(Ss[i], Cs[i], Ss[j], Cs[j], B, min_count, min_groups)
    return weights_from_rho(rho, include_self), rho
