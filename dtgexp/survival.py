"""Weibull life distribution: maximum-likelihood fit (with optional right-censoring) and a profile-likelihood
confidence interval for the shape parameter."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import chi2


def _nll(beta, eta, t, d):
    z = (t / eta) ** beta
    return -(np.sum(d * (np.log(beta) - np.log(eta) + (beta - 1) * np.log(t / eta))) - np.sum(z))


def fit_weibull(t, d=None):
    """Maximum-likelihood (beta, eta) of a Weibull distribution. t: lives (or censoring times); d: 1 = failure,
    0 = right-censored (default: all failures)."""
    t = np.asarray(t, float)
    d = np.ones_like(t) if d is None else np.asarray(d, float)

    def nll(p):
        b, e = np.exp(p)
        return _nll(b, e, t, d)

    best = min((minimize(nll, [np.log(b0), np.log(np.mean(t))], method="Nelder-Mead",
                         options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000}) for b0 in (1.0, 2.0, 4.0)),
               key=lambda r: r.fun)
    return tuple(np.exp(best.x))


def shape_profile_ci(t, level: float = 0.95):
    """Profile-likelihood confidence interval of the Weibull shape beta (all lives observed failures)."""
    t = np.asarray(t, float)
    d = np.ones_like(t)
    beta, eta = fit_weibull(t)
    l0 = _nll(beta, eta, t, d)
    c = chi2.ppf(level, 1) / 2.0

    def excess(b):                                    # eta maximizing the likelihood at a given beta
        return _nll(b, np.mean(t ** b) ** (1.0 / b), t, d) - l0 - c

    return brentq(excess, 1e-3, beta), brentq(excess, beta, 50.0)
