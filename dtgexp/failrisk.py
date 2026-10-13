"""Probability that a CT X-ray tube fails within H days (H = 7 and 30), predicted once per day from routine tube logs.

prior     reliability physics: Weibull wear-out life in scans, fitted in every fold on the lives of the training
          (and validation) tubes. The scans expected in the next H days come from the tube's own workload over the
          last 7 days (causal): P_H = 1 - exp(-[((a + w H) / eta)^beta - (a / eta)^beta])
network   multi-branch correction network: CNN branch on the last `seq_days` days of daily inputs, FNN branch and
          tanh physics-context branch on the last day + the physics context (age, workload, prior logits). The
          network outputs corrections of the prior on the logit scale:
          P30 = sigmoid(logit P30_prior + c30),  P7 = P30 * sigmoid(logit(P7_prior / P30_prior) + c7)
          so that P7 <= P30 always.
loss      binary cross-entropy at both horizons + monotonicity penalty (wear-out: for days of the same tube with a
          similar workload the predicted probability must not fall with age) + L2 penalty on the corrections
          (stay at the physics prior unless the data support a deviation).
federated FedAvg, DTG-v2 (sample-size weights modulated by the label-free, stage-aligned trend similarity), the
          original DTG weighting, FedProx, SCAFFOLD, FedBN, clustered FL and client-level DP-FedAvg; centralized
          training and a data-only network (no prior, no physics loss) as references.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hi import sliced_filament_ratio
from .models import norm1d
from .trends import dtg_weights_pairwise, hi_rate_terms, stage_sums

CUMULATIVE = ("cumulative_thermal_load", "straton_cumul_stress")
EPS = 1e-6


# ------------------------------------------------------------------ physics prior
def weibull_prior(age, rate, H, beta, eta):
    a = np.asarray(age, float) / eta
    b = (np.asarray(age, float) + np.asarray(rate, float) * H) / eta
    return 1.0 - np.exp(-(b ** beta - a ** beta))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ multi-branch correction network
class FailRiskNet(nn.Module):
    """CNN branch on the input sequence (C features x seq_days), FNN branch and tanh physics-context branch on the
    last day and the physics context (E values); fusion of 64 + 64 + 64 + 32 = 224 features; head with two outputs,
    the logit corrections (c7, c30)."""

    def __init__(self, C: int, E: int, norm: str = "bn", p_fnn: float = 0.3, p_fus: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(C, 64, 3, padding=1), norm1d(norm, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), norm1d(norm, 128), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), norm1d(norm, 64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fnn = nn.Sequential(
            nn.Linear(C + E, 128), nn.ReLU(), norm1d(norm, 128), nn.Dropout(p_fnn),
            nn.Linear(128, 128), nn.ReLU(), norm1d(norm, 128), nn.Dropout(p_fnn),
            nn.Linear(128, 64), nn.ReLU(), norm1d(norm, 64), nn.Dropout(p_fnn))
        self.pg = nn.Sequential(nn.Linear(C + E, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 64))
        self.phys = nn.Linear(64, 32)
        self.fusion = nn.Sequential(nn.Linear(64 + 64 + 96, 128), norm1d(norm, 128), nn.ReLU(), nn.Dropout(p_fus))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, ctx):
        last = torch.cat([x[:, -1, :], ctx], 1)
        h = self.pg(last)
        z = self.fusion(torch.cat([self.cnn(x.transpose(1, 2)).squeeze(-1), self.fnn(last), h,
                                   torch.tanh(self.phys(h))], 1))
        return self.head(z)


def probs(out, b, use_prior: bool):
    c7, c30 = out[:, 0], out[:, 1]
    z30 = b["lp30"] + c30 if use_prior else c30
    z7 = b["lr7"] + c7 if use_prior else c7
    p30 = torch.sigmoid(z30)
    return p30 * torch.sigmoid(z7), p30


# ------------------------------------------------------------------ training
@dataclass
class RiskCfg:
    use_prior: bool = True            # False = 'nophys': data-only network, no prior, no physics loss
    fl: str = "fedavg"                # fedavg | dtg | central | fedprox | scaffold | fedbn | cfl | cdp
    rounds: int = 20                  # maximum number of rounds (central: rounds x local_epochs epochs)
    min_rounds: int = 0               # early stopping: never stop before this many rounds
    patience: int = 0                 # early stopping: stop after this many rounds without a validation improvement
    min_rel_improve: float = 1e-3     # (0 = fixed number of rounds); an improvement must lower the loss by 0.1 %
    local_epochs: int = 1
    batch: int = 256
    lr: float = 1e-3
    lam_mono: float = 0.5
    lam_reg: float = 0.01
    mono_dage: float = 0.02           # min age gap (fraction of eta) for a monotonicity pair
    mono_drate: float = 0.3           # max |log workload| gap for a monotonicity pair
    mu: float = 0.01                  # FedProx proximal weight
    scaffold_lr: float = 0.05         # SCAFFOLD local SGD step size
    cfl_warmup: int = 5               # CFL: rounds of FedAvg before the one-shot bipartition
    cdp_z: float = 1.0                # client-level DP-FedAvg noise multiplier
    cdp_clip: float = 2.0             # client-level DP-FedAvg L2 clip of each client update
    device: str = "cpu"
    seed: int = 0


def loss_fn(net, cfg, b, prox=None):
    out = net(b["x"], b["ctx"])
    p7, p30 = probs(out, b, cfg.use_prior)
    loss = (F.binary_cross_entropy(p7.clamp(EPS, 1 - EPS), b["y7"]) +
            F.binary_cross_entropy(p30.clamp(EPS, 1 - EPS), b["y30"]))
    if cfg.use_prior:
        A, R, G = b["a_eta"], b["lrate"], b["tube"]
        m = ((A[None, :] - A[:, None]) > cfg.mono_dage) & ((R[None, :] - R[:, None]).abs() < cfg.mono_drate) & \
            (G[None, :] == G[:, None])
        if m.any():
            pen = F.relu(p30[:, None] - p30[None, :])[m].mean() + F.relu(p7[:, None] - p7[None, :])[m].mean()
            loss = loss + cfg.lam_mono * pen
        loss = loss + cfg.lam_reg * (out ** 2).mean()
    if prox is not None:
        loss = loss + 0.5 * cfg.mu * sum(((p - q) ** 2).sum() for p, q in zip(net.parameters(), prox))
    return loss


def batch_of(T, idx):
    return {k: v[idx] for k, v in T.items()}


def _adapt_bn(base, T, upto, bs=4096):
    """Copy of base whose BatchNorm statistics are re-estimated on windows [0, upto) of this tube
    (label-free AdaBN)."""
    net = copy.deepcopy(base)
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.eval()
    for m in bns:
        m.train()
    for s in range(0, upto, bs):
        idx = torch.arange(s, min(s + bs, upto), device=T["y7"].device)
        if len(idx) > 1:
            net(T["x"][idx], T["ctx"][idx])
    net.eval()
    return net


CAUSAL_FIRST = 10                     # windows (days) of the first reference period of test-time adaptation


def causal_starts(n, chunk=100, first=CAUSAL_FIRST):
    """Chunk starts for strictly causal test-time adaptation: 0, first, 2 first, 4 first, ... up to `chunk`, then every
    `chunk` windows. The windows of a chunk [s_i, s_i+1) are predicted with statistics of the windows [0, s_i) only;
    the first chunk (the first `first` windows) uses its own windows, the reference period at the start of the log."""
    out, s = [0], first
    while s < min(chunk, n):
        out.append(s)
        s *= 2
    out += list(range(chunk, n, chunk))
    return sorted(set(x for x in out if x < n))


@torch.no_grad()
def predict(net, cfg, T, bs=4096, bn_adapt=False, chunk=100, warm=CAUSAL_FIRST):
    """bn_adapt: STRICTLY CAUSAL test-time BN statistics. Windows are in time order; the windows of each chunk are
    predicted with BN statistics estimated on the windows before the chunk (causal_starts); only the first `warm`
    windows (the reference period at the start of the log) use their own statistics. Statistics over later windows
    would carry information about the position of a day in the tube's life."""
    n = T["y7"].shape[0]
    p7s, p30s = np.zeros(n, np.float32), np.zeros(n, np.float32)
    starts = causal_starts(n, chunk, warm) if bn_adapt else [0]
    for i, st in enumerate(starts):
        en = (starts[i + 1] if i + 1 < len(starts) else n) if bn_adapt else n
        m = _adapt_bn(net, T, min(max(st, warm), n)) if bn_adapt else net
        m.eval()
        for s in range(st, en, bs):
            b = batch_of(T, torch.arange(s, min(s + bs, en), device=T["y7"].device))
            p7, p30 = probs(m(b["x"], b["ctx"]), b, cfg.use_prior)
            p7s[s:min(s + bs, en)] = p7.cpu().numpy()
            p30s[s:min(s + bs, en)] = p30.cpu().numpy()
    return p7s, p30s


def bce_np(p, y):
    return -np.mean(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))


def train_epochs(net, T, cfg, epochs, gen, opt, prox=None, ccorr=None):
    """ccorr: SCAFFOLD gradient correction (c - c_k) per parameter. Returns the number of steps."""
    net.train()
    n, steps = T["y7"].shape[0], 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, cfg.batch):
            idx = perm[s:s + cfg.batch].to(T["y7"].device)
            if len(idx) < 2:
                continue
            loss = loss_fn(net, cfg, batch_of(T, idx), prox)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ccorr is not None:
                for p, c in zip(net.parameters(), ccorr):
                    if p.grad is not None:
                        p.grad.add_(c)
            opt.step()
            steps += 1
    return steps


def aggregate(states, w, keys=None):
    out = dict(states[0])
    for k in (keys if keys is not None else states[0].keys()):
        S = torch.stack([s[k].float() for s in states])
        v = (S * torch.as_tensor(w, dtype=torch.float32, device=S.device).view(-1, *[1] * (S.dim() - 1))).sum(0)
        out[k] = v.to(states[0][k].dtype)
    return out


def bn_keys(net):
    keys = set()
    for name, m in net.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            keys |= {f"{name}.{k}" for k in m.state_dict().keys()}
    return keys


class RiskModel:
    """Trained predictor for a held-out tube. CFL picks the cluster by label-free input statistics of the tube's
    earlier days; FedBN and client-level DP use the held-out tube's own (label-free) BN statistics of its earlier
    days. Both are strictly causal (causal_starts): no day is predicted with information from later days."""

    def __init__(self, cfg, states, centroids=None):
        self.cfg, self.states, self.centroids = cfg, states, centroids

    def predict(self, net, T, feat_mean=None, chunk=100):
        bn_adapt = self.cfg.fl in ("fedbn", "cdp")
        if self.centroids is None:
            net.load_state_dict(self.states[0])
            return predict(net, self.cfg, T, bn_adapt=bn_adapt, chunk=chunk)
        n = T["y7"].shape[0]                         # CFL: cluster from the mean inputs of the days before each chunk
        p7s, p30s = np.zeros(n, np.float32), np.zeros(n, np.float32)
        starts = causal_starts(n, chunk)
        for i, st in enumerate(starts):
            en = starts[i + 1] if i + 1 < len(starts) else n
            upto = min(max(st, CAUSAL_FIRST), n)
            fm = T["x"][:upto, -1, :].float().mean(0).cpu().numpy()
            k = int(np.argmin([np.linalg.norm(fm - c) for c in self.centroids]))
            net.load_state_dict(self.states[k])
            part = {kk: v[st:en] for kk, v in T.items() if torch.is_tensor(v) and v.shape[:1] == (n,)}
            a7, a30 = predict(net, self.cfg, part, bn_adapt=bn_adapt, chunk=chunk)
            p7s[st:en], p30s[st:en] = a7, a30
        return p7s, p30s


def _flat(sd, names):
    return torch.cat([sd[n].float().reshape(-1) for n in names])


def train(clients: List[Dict], V: Optional[List[Dict]], cfg: RiskCfg, C: int, E: int, weights=None):
    """clients: one tensor dict per training tube (each with 'feat_mean', a label-free input summary).
    V: list of validation tubes (tensor dicts). Returns (RiskModel, validation curve, info)."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    gnet = FailRiskNet(C, E).to(dev)
    clone = lambda sd: {k: v.detach().clone() for k, v in sd.items()}
    pnames = [k for k, _ in gnet.named_parameters()]
    bnk = bn_keys(gnet)
    strip = lambda T: {k: v for k, v in T.items() if torch.is_tensor(v)}
    clients_t = [strip(c) for c in clients]
    info = {"clusters": None}

    def val_of(model):
        if not V:
            return float("nan")
        tot = 0.0
        for v in V:
            p7, p30 = model.predict(gnet_eval, strip(v), v.get("feat_mean"), chunk=500)
            tot += bce_np(p7, v["y7"].cpu().numpy()) + bce_np(p30, v["y30"].cpu().numpy())
        return tot / len(V)

    gnet_eval = copy.deepcopy(gnet)
    best, curve = (float("inf"), None), []
    es = {"best": float("inf"), "at": -1}

    def keep(v, m):
        """Best model by validation loss (without validation tubes: the latest model)."""
        nonlocal best
        if best[1] is None or not V or (np.isfinite(v) and (v < best[0] or not np.isfinite(best[0]))):
            best = (v, m)

    def stop(i, v, per_round=1):
        """Early stopping after evaluation i (0-based; central evaluates every epoch = 1/per_round round): no
        improvement of at least min_rel_improve for `patience` rounds, and at least `min_rounds` rounds done."""
        if np.isfinite(v) and v < es["best"] * (1.0 - cfg.min_rel_improve):
            es["best"], es["at"] = v, i
        if cfg.patience <= 0 or not np.isfinite(v):
            return False
        return i + 1 >= cfg.min_rounds * per_round and i - es["at"] >= cfg.patience * per_round

    if cfg.fl == "central":
        T = {k: torch.cat([c[k] for c in clients_t]) for k in clients_t[0]}
        opt = torch.optim.Adam(gnet.parameters(), lr=cfg.lr)
        for ep in range(cfg.rounds * cfg.local_epochs):
            train_epochs(gnet, T, cfg, 1, gen, opt)
            m = RiskModel(cfg, [clone(gnet.state_dict())])
            v = val_of(m)
            curve.append(v)
            keep(v, m)
            if stop(ep, v, cfg.local_epochs):
                break
        info["rounds_run"] = len(curve) / cfg.local_epochs
        return best[1], curve, info

    K = len(clients_t)
    n = np.array([c["y7"].shape[0] for c in clients_t], float)
    w_all = (n / n.sum()) if weights is None else np.asarray(weights, float) / np.sum(weights)
    if cfg.fl == "cdp":
        w_all = np.ones(K) / K
    nets = [copy.deepcopy(gnet) for _ in clients_t]
    if cfg.fl == "scaffold":
        opts = [torch.optim.SGD(m.parameters(), lr=cfg.scaffold_lr) for m in nets]
        c_glob = [torch.zeros_like(p) for p in gnet.parameters()]
        c_loc = [[torch.zeros_like(p) for p in gnet.parameters()] for _ in clients_t]
    else:
        opts = [torch.optim.Adam(m.parameters(), lr=cfg.lr) for m in nets]
    clusters = [list(range(K))]
    gstates = [clone(gnet.state_dict())]
    feat = np.stack([c["feat_mean"] for c in clients])
    for r in range(cfg.rounds):
        new_states = []
        for ci, members in enumerate(clusters):
            g = gstates[ci]
            local, steps = [], []
            for k in members:
                m = nets[k]
                if cfg.fl == "fedbn" and r > 0:
                    sd = m.state_dict()
                    m.load_state_dict({kk: (sd[kk] if kk in bnk else g[kk]) for kk in g})
                else:
                    m.load_state_dict(g)
                prox = [g[pn].detach() for pn in pnames] if cfg.fl == "fedprox" else None
                ccorr = [cg - cl for cg, cl in zip(c_glob, c_loc[k])] if cfg.fl == "scaffold" else None
                steps.append(train_epochs(m, clients_t[k], cfg, cfg.local_epochs, gen, opts[k], prox, ccorr))
                local.append(clone(m.state_dict()))
            w = w_all[members] / w_all[members].sum()
            if cfg.fl == "cdp":                       # clip, average uniformly, Gaussian noise; buffers not shared
                deltas = []
                for sd in local:
                    d = _flat(sd, pnames) - _flat(g, pnames)
                    deltas.append(d * min(1.0, cfg.cdp_clip / (float(d.norm()) + 1e-12)))
                avg = torch.stack(deltas).mean(0)
                avg = avg + torch.randn(avg.shape, generator=gen).to(avg.device) * cfg.cdp_z * cfg.cdp_clip / len(members)
                ng, off = dict(g), 0
                for pn in pnames:
                    k_ = g[pn].numel()
                    ng[pn] = (g[pn].float() + avg[off:off + k_].view_as(g[pn])).to(g[pn].dtype)
                    off += k_
            else:
                ng = aggregate(local, w)
                if cfg.fl == "scaffold":
                    for j, k in enumerate(members):
                        for i, pn in enumerate(pnames):
                            c_new = c_loc[k][i] - c_glob[i] + (g[pn] - local[j][pn]) / (max(steps[j], 1) * cfg.scaffold_lr)
                            dc = c_new - c_loc[k][i]
                            c_loc[k][i] = c_new
                            c_glob[i] = c_glob[i] + dc / K
            new_states.append(ng)
            if cfg.fl == "cfl" and r + 1 == cfg.cfl_warmup and len(clusters) == 1 and K >= 4:
                from scipy.cluster.hierarchy import fcluster, linkage
                D = torch.stack([_flat(sd, pnames) - _flat(g, pnames) for sd in local]).cpu().numpy()
                Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
                cos = np.clip(Dn @ Dn.T, -1, 1)
                lab = fcluster(linkage(1 - cos[np.triu_indices(K, 1)], "complete"), 2, "maxclust")
                parts = [[members[j] for j in range(len(members)) if lab[j] == c] for c in (1, 2)]
                if min(len(p) for p in parts) >= 2:
                    clusters = parts
                    new_states = [ng, clone(ng)]
                    info["clusters"] = [list(map(int, p)) for p in parts]
        gstates = new_states
        cents = [feat[c].mean(0) for c in clusters] if len(clusters) > 1 else None
        model = RiskModel(cfg, [clone(s) for s in gstates], cents)
        v = val_of(model)
        curve.append(v)
        keep(v, model)
        if stop(r, v):
            break
    info["rounds_run"] = len(curve)
    if cfg.fl == "cdp":                               # privacy spent by every round actually run (no noise: none)
        from .dp import epsilon
        info["cdp_epsilon"] = float(epsilon(1.0, cfg.cdp_z, len(curve), 1e-3)) if cfg.cdp_z > 0 else float("inf")
    return best[1], curve, info


# ------------------------------------------------------------------ DTG-v2 (label-free, stage-aligned trend)
def hi_stage_trend(d: Dict, D: float, fine_bins: int = 100, dp: Optional[Dict] = None):
    """Trend of one client: degradation rate of its filament HI per HI stage, as per-bin sums and counts.
    dp = {"eps", "r_clip", "lag", "rng"}: unsmoothed, clipped terms + Laplace mechanism."""
    hi = 1.0 - (1.0 - d["hi"].astype(float)) / D
    if dp:
        rate, level = hi_rate_terms(hi, d["sc"] / 1000.0, smooth=1, lag=dp["lag"], r_clip=dp["r_clip"])
    else:
        rate, level = hi_rate_terms(hi, d["sc"])
    if not len(rate):
        return np.zeros(fine_bins), np.zeros(fine_bins)
    s, c = stage_sums(rate, np.clip(level, 0, 1), fine_bins)
    if dp:
        from .dp import privatize_trend
        s, c = privatize_trend(s, c, dp["eps"], dp["r_clip"], dp["rng"])
    return s, c


def eol_decline(tubes: List[Dict]) -> float:
    declines = []
    for d in tubes:
        h = d["hi"][np.isfinite(d["hi"])]
        if len(h) > 10:
            declines.append(max(1.0 - float(np.median(h[-max(10, len(h) // 50):])), 1e-4))
    return float(np.median(declines)) if declines else 0.05


def dtg_weights_for(tubes: List[Dict], fine_bins: int = 100, B: int = 20, dp: Optional[Dict] = None, D=None):
    """Label-free stage trend of every client, pairwise Pearson correlation over the shared stage range, and the
    original DTG weights (each client's mean similarity, normalized). Returns (weights, correlation matrix)."""
    D = eol_decline(tubes) if D is None else D
    Ss, Cs = zip(*[hi_stage_trend(d, D, fine_bins, dp) for d in tubes])
    return dtg_weights_pairwise(list(Ss), list(Cs), B=B)


def dtg2_weights(rho, n, tau: float = 0.05, perm=None):
    """DTG-v2: FedAvg sample-size weights modulated by the trend similarity. s_k = mean over j != k of (rho_kj + 1) / 2 (own similarity excluded);
    w_k proportional to n_k * exp((s_k - mean(s)) / tau). Identical similarities give exactly FedAvg.
    perm (control experiment): s is permuted across the clients before weighting, so the weights keep their
    spread but lose the link between a client and its own trend. Returns (weights, s as used)."""
    R = np.asarray(rho, float)
    K = len(R)
    S = np.clip((R + 1.0) / 2.0, 0.0, 1.0)
    s = (S.sum(1) - np.diag(S)) / max(K - 1, 1)
    if perm is not None:
        s = s[np.asarray(perm, int)]
    g = np.exp(np.clip((s - s.mean()) / tau, -50.0, 50.0))
    w = np.asarray(n, float) * g
    return w / w.sum(), s


def dtg_weights_index(tubes: List[Dict], D=None, Lmax: int = 200):
    """Index-based trend of the original manuscript (equispaced subsampling over the whole log and pairwise
    interpolation to a common length), used only for the truncation comparison."""
    from .trends import weights_from_rho
    D = eol_decline(tubes) if D is None else D
    V = []
    for d in tubes:
        h = 1.0 - (1.0 - d["hi"].astype(float)) / D
        h = h[np.isfinite(h)]
        V.append(h[np.linspace(0, len(h) - 1, min(len(h), Lmax)).astype(int)] if len(h) else np.zeros(2))
    K = len(V)
    rho = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            L = min(len(V[i]), len(V[j]))
            a = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(V[i])), V[i])
            b = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(V[j])), V[j])
            r = np.corrcoef(a, b)[0, 1] if L > 2 and a.std() > 0 and b.std() > 0 else 0.0
            rho[i, j] = rho[j, i] = r
    return weights_from_rho(rho), rho


# ------------------------------------------------------------------ metrics
def tube_weights(tube_ids: np.ndarray) -> np.ndarray:
    u, inv, cnt = np.unique(tube_ids, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def metrics(y, p, w):
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {"auc": np.nan, "ap": np.nan, "brier": float(np.average((p - y) ** 2, weights=w))}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p, sample_weight=w))
        out["ap"] = float(average_precision_score(y, p, sample_weight=w))
    return out


def bootstrap_delta(df: pd.DataFrame, a: str, b: str, ycol: str, reps: int = 2000, seed: int = 0):
    """Difference in tube-weighted AUC (column a minus column b): the estimate on all tubes, with the 2.5th and 97.5th
    percentiles of a tube-level bootstrap as the 95% CI and the bootstrap share of differences <= 0."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    w_all = np.concatenate([np.full(len(g), 1.0 / len(g)) for _, g in df.groupby("tube")])
    srt = pd.concat([g for _, g in df.groupby("tube")])
    full = roc_auc_score(srt[ycol], srt[a], sample_weight=w_all) - roc_auc_score(srt[ycol], srt[b], sample_weight=w_all)
    groups = [g for _, g in df.groupby("tube")]
    deltas = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        parts = [groups[i] for i in pick]
        y = np.concatenate([g[ycol].to_numpy() for g in parts])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        w = np.concatenate([np.full(len(g), 1.0 / len(g)) for g in parts])
        pa = np.concatenate([g[a].to_numpy() for g in parts])
        pb = np.concatenate([g[b].to_numpy() for g in parts])
        deltas.append(roc_auc_score(y, pa, sample_weight=w) - roc_auc_score(y, pb, sample_weight=w))
    d = np.array(deltas)
    return float(full), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))


def alarm_stats(df: pd.DataFrame, pcol: str, thr: float, H: float = 30.0):
    """Per tube: lead time (days before failure from which the probability stays >= thr) and the
    share of windows outside the last H days on which the alarm is on (false-alarm time, %)."""
    leads, fas = [], []
    for _, g in df.sort_values("dtf_days", ascending=False).groupby("tube"):
        above = g[pcol].to_numpy() >= thr
        dtf = g["dtf_days"].to_numpy()
        lead = 0.0
        if above[-1]:
            k = len(above) - 1
            while k > 0 and above[k - 1]:
                k -= 1
            lead = float(dtf[k])
        leads.append(lead)
        out = dtf > H
        fas.append(100.0 * above[out].mean() if out.any() else 0.0)
    return float(np.median(leads)), float(np.mean([l > 0 for l in leads])), float(np.median(fas))


# ------------------------------------------------------------------ daily inputs
def _daily(v, day, n_days, how="mean"):
    s = pd.Series(np.asarray(v, float)).groupby(day)
    s = s.mean() if how == "mean" else s.sum()
    return s.reindex(range(n_days)).to_numpy()


def _rolling_slope(y, w):
    """Causal least-squares slope of a daily series over the last w days (per day)."""
    out = np.full(len(y), np.nan)
    for d in range(len(y)):
        v = y[max(0, d - w + 1):d + 1]
        x = np.arange(len(v), dtype=float)
        m = np.isfinite(v)
        if m.sum() >= 3:
            xm, vm = x[m] - x[m].mean(), v[m] - v[m].mean()
            den = (xm ** 2).sum()
            out[d] = (xm * vm).sum() / den if den > 0 else 0.0
    return out


# ------------------------------------------------------------------ daily inputs A2 (engineered) and B2 (+ precursors)
# Construction (fixed by a feature audit, independent of model results):
#  1 reference period = the first `ref_days` ACTIVE days (>= `active_min` scans) on which the feature is valid,
#    not the first calendar days (logs can start with a few acceptance-test days and months of no use);
#  2 features that cannot be computed yet stay missing until valid; before the first valid value the input is
#    0 = "at reference";
#  3 offset from the tube's own reference only (no division by the tube's own spread); scaling is done later
#    with training-tube statistics (robust, then clipped);
#  4 aborts = scans ended almost immediately (actual < 20 % of nominal scan time), the end-of-life signature in
#    the tube records (operator-stopped scans end later); the abort-reason field is absent in some logs and is
#    not used;
#  5 arcing rates on a log scale (heavy-tailed counts).
PRECURSORS_V2 = ("arcing_log_per_1000", "arcing_log_per_1000_7d", "immediate_abort_share", "immediate_abort_share_7d",
                 "kv_deviation", "ma_deviation", "ma_spread", "filament_hi", "filament_slope_7d", "filament_slope_30d",
                 "filament_accel", "oil_residual", "oil_residual_trend_14d")
IMMEDIATE_ABORT = 0.2


def reference_offset(M, active, ref_days=30):
    """Per column: median over the first `ref_days` active days on which the column is valid (expanding,
    causal). Returns M - reference, with 0 where the column (or its reference) is not yet available."""
    n, F = M.shape
    ref = np.full((n, F), np.nan)
    for j in range(F):
        vals, cur = [], np.nan
        col = M[:, j]
        for d in range(n):
            if active[d] and len(vals) < ref_days and np.isfinite(col[d]):
                vals.append(col[d])
                cur = float(np.median(vals))
            ref[d, j] = cur
    Z = M - ref
    Z[~np.isfinite(Z)] = 0.0
    return Z.astype(np.float32)


def daily_precursors_v2(df, day, n_days, hi, active, ref_days=30):
    """Precursors per day, causal; values stay missing until they can be computed. Returns
    (matrix with the PRECURSORS_V2 columns, missing-field notes)."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) if c in df.columns else None
    cnt = np.bincount(day, minlength=n_days).astype(float)
    roll = lambda v, w: pd.Series(v).rolling(w, min_periods=1).sum().to_numpy()
    got, missing = {}, []
    a1, a2 = num("arcings_one"), num("arcings_both")
    if a1 is not None or a2 is not None:
        arc = np.nan_to_num(a1 if a1 is not None else 0.0) + np.nan_to_num(a2 if a2 is not None else 0.0)
        per_day = np.nan_to_num(_daily(arc, day, n_days, "sum"))
        got["arcing_log_per_1000"] = np.where(cnt > 0, np.log1p(1000 * per_day / np.maximum(cnt, 1)), np.nan)
        got["arcing_log_per_1000_7d"] = np.log1p(1000 * roll(per_day, 7) / np.maximum(roll(cnt, 7), 1))
    else:
        missing.append("arcing")
    sa, sn = num("scan_time_act"), num("scan_time_nom")
    if sa is not None and sn is not None:
        ab = ((sn > 0) & (sa < IMMEDIATE_ABORT * sn)).astype(float)
        ab_day = np.nan_to_num(_daily(ab, day, n_days, "sum"))
        got["immediate_abort_share"] = np.where(cnt > 0, ab_day / np.maximum(cnt, 1), np.nan)
        got["immediate_abort_share_7d"] = roll(ab_day, 7) / np.maximum(roll(cnt, 7), 1)
    else:
        missing.append("aborts")
    va, vn = num("voltage_act"), num("voltage_nom")
    if va is not None and vn is not None:
        got["kv_deviation"] = _daily(np.abs(va - vn) / np.where(vn > 0, vn, np.nan), day, n_days)
    else:
        missing.append("kV deviation")
    cm, cn, cmin, cmax = num("current_mean"), num("current_nom"), num("current_min"), num("current_max")
    if cm is not None and cn is not None:
        den = np.where(cn > 0, cn, np.nan)
        got["ma_deviation"] = _daily(np.abs(cm - cn) / den, day, n_days)
        if cmin is not None and cmax is not None:
            got["ma_spread"] = _daily((cmax - cmin) / den, day, n_days)
    else:
        missing.append("mA deviation")
    h = pd.Series(_daily(hi, day, n_days)).ffill().to_numpy()
    if np.isfinite(h).any():
        s7, s30 = _rolling_slope(h, 7), _rolling_slope(h, 30)
        got.update({"filament_hi": h, "filament_slope_7d": s7, "filament_slope_30d": s30, "filament_accel": s7 - s30})
    else:
        missing.append("filament HI")
    ob, oe, gt = num("oil_temp_begin"), num("oil_temp_end"), num("gantry_temp")
    if ob is not None and oe is not None and gt is not None and va is not None and cm is not None and sa is not None:
        A = np.column_stack([va * cm * sa / 1000.0, ob - gt, np.ones(len(ob))])
        y = oe - ob
        ok = np.isfinite(A).all(1) & np.isfinite(y)
        ref_set = np.where(active)[0][:ref_days]
        ref_end = int(ref_set[-1]) if len(ref_set) else -1
        inref = np.isin(day, ref_set)
        res = np.full(len(y), np.nan)
        coef = None
        for d in range(n_days):            # fit on earlier reference days only, fixed after the reference period
            if d <= ref_end + 1:
                m = ok & inref & (day < d)
                if m.sum() >= 50:
                    coef = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
            if coef is not None:
                sel = ok & (day == d)
                res[sel] = y[sel] - A[sel] @ coef
        r = pd.Series(_daily(res, day, n_days)).ffill().to_numpy()
        got["oil_residual"], got["oil_residual_trend_14d"] = r, _rolling_slope(r, 14)
    else:
        missing.append("oil residual")
    P = np.column_stack([np.asarray(got.get(c, np.full(n_days, np.nan)), float) for c in PRECURSORS_V2])
    P = pd.DataFrame(P).ffill().to_numpy()     # carry the last value over days without scans; leading gaps stay missing
    return P, missing


def build_daily_v2(path: str, fe, kind: str = "A", seq_days: int = 60, ref_days: int = 30, active_min: int = 20,
                   scan: Optional[Dict] = None) -> Dict:
    """Daily inputs (see the notes above PRECURSORS_V2). One sample per day; X = the last `seq_days` days, each
    feature as an offset from the tube's own reference period (causal). kind 'A': daily means of the engineered
    features; 'B': A + failure precursors. fe = the feature module (features/tube_features.py). scan: optional
    per-scan cache {X, t, sc, hi, features} of the same log (reused when it has the same number of records)."""
    df = fe.load_raw_csv(path)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if scan is not None and len(scan["X"]) == len(df):
        X, t, sc, hi, names = scan["X"], scan["t"].astype(float), scan["sc"], scan["hi"], list(scan["features"])
    else:
        feats = fe.compute_all_features(df)
        feats = feats.drop(columns=[c for c in CUMULATIVE if c in feats.columns])
        X = feats.to_numpy(np.float32)
        X[~np.isfinite(X)] = 0.0
        names = list(feats.columns)
        t = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        t = t.astype(np.float32).astype(float)        # days since the first record, stored in single precision
        sc = pd.to_numeric(df["scan_count"], errors="coerce").ffill().bfill().to_numpy(float)
        try:
            hi = np.asarray(sliced_filament_ratio(df)[0], np.float32)
        except Exception:
            hi = np.full(len(df), np.nan, np.float32)
    day = np.floor(t).astype(int)
    n_days = int(day.max()) + 1
    cnt = np.bincount(day, minlength=n_days).astype(float)
    active = cnt >= active_min
    D = pd.DataFrame(X).groupby(day).mean().reindex(range(n_days)).ffill().to_numpy()
    M = np.column_stack([D, np.log1p(cnt), (cnt > 0).astype(float)])
    cols = names + ["log_scans_today", "active_day"]
    missing = []
    if kind == "B":
        P, missing = daily_precursors_v2(df, day, n_days, hi, active, ref_days)
        M = np.column_stack([M, P])
        cols += list(PRECURSORS_V2)
    Z = reference_offset(M, active, ref_days)
    t_fail = float(t[-1])
    tau = np.minimum(np.arange(n_days) + 1.0, t_fail)
    sc_end = pd.Series(sc).groupby(day).max().reindex(range(n_days)).ffill().bfill().to_numpy()
    prev = np.concatenate([np.full(7, sc[0]), sc_end])[:n_days]
    span = np.minimum(np.arange(n_days) + 1.0, 7.0)
    rate = np.maximum((sc_end - prev) / span, 1.0)
    Zp = np.vstack([np.repeat(Z[:1], seq_days - 1, 0), Z])
    ref_days_found = int(min(active.sum(), ref_days))
    return {"X": Zp, "features": cols, "t": t.astype(np.float32), "sc": sc, "ends": np.arange(seq_days - 1, seq_days - 1 + n_days),
            "age": sc_end, "rate": rate, "dtf": (t_fail - tau).astype(np.float32), "life": float(sc[-1]), "hi": hi,
            "days": t_fail, "missing": np.array(missing if missing else [""]), "pad": int(seq_days - 1),
            "ref_active_days": ref_days_found}
