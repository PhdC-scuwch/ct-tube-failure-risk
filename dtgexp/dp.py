"""Differential privacy (Sec. IV-E of the paper).

* Client-level DP-FedAvg (McMahan et al., ICLR 2018) is implemented in dtgexp/failrisk.py (fl = "cdp"): every
  client update is clipped to L2 norm S, the clipped updates are averaged with equal weights and Gaussian noise
  with standard deviation z S / K is added to the average. All clients take part in every round (sampling rate 1).
* Accounting: Renyi DP of the (sub)sampled Gaussian mechanism (integer orders), converted with
  eps = rdp + log(1/delta) / (alpha - 1) (Mironov, CSF 2017), over the rounds actually run.
* The label-free trend of DTG-v2 is released once with the Laplace mechanism on its per-bin sums and counts
  (Dwork et al., TCC 2006); see privatize_trend.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

ORDERS = list(range(2, 128)) + [160, 192, 256]


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """Renyi DP of order alpha of the Gaussian mechanism with noise multiplier sigma and sampling rate q."""
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * sigma ** 2)
    terms = []
    for k in range(alpha + 1):
        lc = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        term = lc + (alpha - k) * math.log1p(-q) + (k * math.log(q) if k > 0 else 0.0)
        terms.append(term + (k * k - k) / (2.0 * sigma ** 2))
    return float(np.logaddexp.reduce(terms)) / (alpha - 1)


def epsilon(q: float, sigma: float, steps: int, delta: float, orders: Sequence[int] = ORDERS) -> float:
    """(epsilon, delta)-DP after `steps` compositions."""
    best = float("inf")
    for a in orders:
        best = min(best, steps * rdp_sampled_gaussian(q, sigma, a) + math.log(1.0 / delta) / (a - 1))
    return best


def privatize_trend(sums: np.ndarray, counts: np.ndarray, eps: float, r_clip: float,
                    rng: np.random.Generator):
    """Laplace mechanism on the per-bin (sum, count) of HI-rate terms computed WITHOUT smoothing.
    One record enters at most 2 rate terms (as start and as end of a lag difference); each term lies in
    [-r_clip, r_clip] and may move between bins -> L1 sensitivity 4 r_clip (sums) and 4 (counts).
    eps is split equally between the sums and the counts."""
    b_s = 4.0 * r_clip / (eps / 2.0)
    b_c = 4.0 / (eps / 2.0)
    s = sums + rng.laplace(0.0, b_s, size=sums.shape)
    c = np.maximum(counts + rng.laplace(0.0, b_c, size=counts.shape), 0.0)
    return s, c
