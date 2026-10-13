#!/usr/bin/env python3
"""Probability that a CT X-ray tube fails within 7 and 30 days: training and evaluation of the physics-guided
multi-branch network with federated learning, against the physics prior and the federated baselines.

Variants (same network everywhere; only the aggregation or the physics guidance differs)
  prior            Weibull wear-out life in scans + the tube's recent workload (no learning)
  fedavg           physics-guided network (prior + learned correction, physics-guided loss), FedAvg aggregation
  dtg2[-t<tau>]    the same with DTG-v2 aggregation (label-free stage-aligned trend), temperature tau (default 0.05)
  dtg              the same with the original DTG weighting (mean trend similarity, normalized)
  central          the same trained on pooled data (reference, not federated)
  nophys           data-only multi-branch network: no prior, no physics loss, FedAvg
  fedprox scaffold fedbn cfl     federated baselines with the same network
  fedbn-dtg2       FedBN aggregated with the DTG-v2 weights
  dtg2-dp<eps>     DTG-v2 with its trend released under eps-DP (Laplace mechanism), e.g. dtg2-dp1
  cdp<z>           client-level DP-FedAvg, noise multiplier z (cdp0: the same setting without noise)
  fedavg-eq        control: FedAvg with equal client weights instead of sample-size weights
  dtg2-shuf        control: DTG-v2 with the trend similarities permuted across the clients (fixed per fold and seed)
  <v>-m0           the same without the monotonicity loss (e.g. fedavg-m0)
  <v>-r<lam>       the same with correction penalty lam_reg = lam (e.g. fedavg-r0.1, central-r1)
  fedavg-tuned dtg2-tuned central-tuned   nested selection of lam_reg from --tune-reg per fold on the
                   validation tubes (lowest validation loss); dtg2-tuned chooses the DTG-v2 temperature from
                   --tune-tau at the same time; dtg2-t<tau>-tuned = penalty chosen at a fixed temperature;
                   every grid member is also reported with fixed settings. For the tuned families, the final models
                   <F>-ens (ensemble selection over the grid) and the causal smoothing <x>-s are built on the
                   validation tubes as well (no extra training).
Protocols: --protocol loo (leave one tube out) or loho (leave one hospital out, needs --site-map).
The held-out tube is never trained on. Inside each fold, --val-tubes training tubes are held out to select the round
and to stop training (early stopping, with --patience); the Weibull prior is fitted to the lives of the training and
validation tubes, and the feature scaling uses the training tubes only. --workers N runs the folds in N processes
(same results). --device auto uses the first CUDA GPU if PyTorch sees one, otherwise the CPU.

  python scripts/failrisk.py --raw-dir data/tubes --input daily-B2 --out results/B2/loo --seeds 0,1,2 ...
  (reproduce.py runs every step with the settings of the paper)
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtgexp.cli import resolve_device, setup_console  # noqa: E402
from dtgexp.failrisk import (FailRiskNet, RiskCfg, alarm_stats, bn_keys, bootstrap_delta, build_daily_v2,  # noqa: E402
                             PRECURSORS_V2,
                             dtg2_weights, dtg_weights_for, dtg_weights_index, eol_decline, logit, metrics, train,
                             tube_weights, weibull_prior)
from dtgexp.survival import fit_weibull  # noqa: E402

BASE = {"fedavg": "fedavg", "dtg": "dtg", "central": "central", "nophys": "fedavg", "fedprox": "fedprox",
        "scaffold": "scaffold", "fedbn": "fedbn", "cfl": "cfl"}


def atomic_csv(df, fn):
    tmp = fn + ".tmp"
    df.to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, fn)


def atomic_json(obj, fn):
    tmp = fn + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, fn)


def read_done(fn, tubes, tests):
    """A saved fold is reused only if it is complete: right number of rows, no missing predictions."""
    try:
        df = pd.read_csv(fn)
    except Exception:
        return None
    need = sum(len(tubes[t]["ends"]) for t in tests)
    if len(df) != need or not {"p7", "p30", "tube"} <= set(df.columns) or df[["p7", "p30"]].isna().any().any():
        return None
    return df


def parse_variant(v):
    """-> (fl, use_prior, extra). Optional suffix -r<lambda> sets the correction penalty lam_reg (e.g. fedavg-r0.1,
    dtg2-r1), -m<lambda> the weight of the monotonicity loss (fedavg-m0: without it); extra also carries dp_eps
    (dtg-dp<eps>, dtg2-dp<eps>), cdp_z (cdp<z>) and dtg2_tau (dtg2[-t<tau>])."""
    m = re.fullmatch(r"(.+)-m([0-9.]+)", v)
    if m:
        fl, use_prior, extra = parse_variant(m.group(1))
        return fl, use_prior, {**extra, "lam_mono": float(m.group(2))}
    m = re.fullmatch(r"(.+)-r([0-9.]+)", v)
    if m:
        fl, use_prior, extra = _parse_base(m.group(1))
        return fl, use_prior, {**extra, "lam_reg": float(m.group(2))}
    return _parse_base(v)


DTG2_TAU = 0.05                                     # DTG-v2 temperature of the plain name "dtg2"


def dtg2_name(tau):
    return "dtg2" if abs(tau - DTG2_TAU) < 1e-12 else f"dtg2-t{tau:g}"


def is_tuned(v):
    return v.endswith("-tuned")


def tune_members(tuned, a):
    """Grid members of a tuned variant <base>-tuned; per fold the member with the lowest validation loss is used.
      fedavg-tuned, central-tuned, ...  one member per correction penalty in --tune-reg (the default penalty keeps
                                        the plain name: fedavg, fedavg-r0.1, fedavg-r1)
      dtg2-tuned                        every DTG-v2 temperature in --tune-tau x every penalty (dtg2-t0.02-r0.1, ...)
      dtg2-t<tau>-tuned                 the penalties at that temperature only (how the result depends on tau)"""
    base = tuned[:-len("-tuned")]
    lams = [float(x) for x in a.tune_reg.split(",") if x.strip()] or [a.lam_reg]
    if base == "dtg2":
        taus = [float(x) for x in a.tune_tau.split(",") if x.strip()] or [DTG2_TAU]
        bases = list(dict.fromkeys(dtg2_name(t) for t in taus))
    else:
        m = re.fullmatch(r"dtg2-t([0-9.]+)", base)
        bases = [dtg2_name(float(m.group(1)))] if m else [base]
    return [b if abs(lam - a.lam_reg) < 1e-12 else f"{b}-r{lam:g}" for b in bases for lam in lams]


def _parse_base(v):
    if v in BASE:
        return BASE[v], v != "nophys", {}
    m = re.fullmatch(r"dtg2(?:-t([0-9.]+))?", v)
    if m:
        return "dtg", True, {"dtg2_tau": float(m.group(1)) if m.group(1) else DTG2_TAU}
    m = re.fullmatch(r"dtg-dp([0-9.]+)", v)
    if m:
        return "dtg", True, {"dp_eps": float(m.group(1))}
    m = re.fullmatch(r"dtg2-dp([0-9.]+)", v)
    if m:
        return "dtg", True, {"dtg2_tau": DTG2_TAU, "dp_eps": float(m.group(1))}
    m = re.fullmatch(r"cdp([0-9.]+)", v)
    if m:
        return "cdp", True, {"cdp_z": float(m.group(1))}
    m = re.fullmatch(r"fedbn-dtg2(?:-t([0-9.]+))?", v)   # FedBN (local BN) aggregated with the DTG-v2 weights
    if m:
        return "fedbn", True, {"dtg2_tau": float(m.group(1)) if m.group(1) else DTG2_TAU}
    if v == "fedavg-eq":                                 # control: FedAvg with equal client weights
        return "fedavg", True, {"equal_w": True}
    m = re.fullmatch(r"dtg2-shuf(?:-t([0-9.]+))?", v)    # control: DTG-v2 with the similarities permuted across clients
    if m:
        return "dtg", True, {"dtg2_tau": float(m.group(1)) if m.group(1) else DTG2_TAU, "shuffle": True}
    raise SystemExit(f"unknown variant {v}")


INPUTS = {"daily-A2": "A", "daily-B2": "B"}           # A2: engineered features; B2: A2 + failure precursors
CACHE_TAG = {"daily-A2": "daily2A", "daily-B2": "daily2B"}
FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")


def read_exclusions(path):
    """tube -> reason, from the data audit's exclusion.csv (columns tube, excluded, reasons)."""
    if not path:
        return {}
    t = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for r in t.itertuples():
        if str(r.excluded).strip().lower() in ("true", "1", "yes"):
            out[str(r.tube).strip()] = str(getattr(r, "reasons", "")).strip()
    return out


def input_scaling(tubes, names, a):
    """Feature scaling from the training tubes: robust (median, 1.4826 MAD, floor 0.1 SD) over the tubes' real
    days, then clipped to +-5."""
    R = np.concatenate([tubes[n]["X"][int(tubes[n].get("pad", 0)):] for n in names])
    mu = np.median(R, 0)
    sd = np.maximum(1.4826 * np.median(np.abs(R - mu), 0), np.maximum(0.1 * R.std(0), 1e-6))
    return mu, sd, 5.0


def load_tubes(a):
    sys.path.insert(0, os.path.abspath(a.src))
    import tube_features as fe
    files = sorted(glob.glob(os.path.join(a.raw_dir, a.pattern)),
                   key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p)])
    cache = a.cache or os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    tubes = {}
    kind = INPUTS[a.input]
    excluded = read_exclusions(a.exclude_file)
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        if name in excluded:
            print(f"{name}: excluded by the data-quality criteria ({excluded[name]})")
            continue
        cp = os.path.join(cache, f"{name}__{CACHE_TAG[a.input]}_L{a.seq_days}.npz")
        d = None
        if os.path.exists(cp):
            try:
                z = np.load(cp, allow_pickle=False)
                d = {k: z[k] for k in z.files}
                d["features"] = list(d["features"])
                for k in ("life", "days"):
                    d[k] = float(d[k])
            except Exception:
                print(f"{name}: cache file damaged (e.g. power loss while writing) - rebuilding")
                d = None
        if d is None:
            with redirect_stdout(io.StringIO()):            # the feature module prints its progress
                d = build_daily_v2(p, fe, kind, a.seq_days)
            tmp = cp[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, **{k: (np.array(v) if k == "features" else v) for k, v in d.items()})
            os.replace(tmp, cp)
        keep = d["dtf"] >= a.min_dtf                # the failure itself must not be in the input
        for k in ("ends", "age", "rate", "dtf"):
            d[k] = d[k][keep]
        if d["life"] < a.min_life or len(d["ends"]) < 20:
            print(f"{name}: life {d['life']:.0f} scans, {len(d['ends'])} windows - excluded (infant failure / too short)")
            continue
        tubes[name] = d
        miss = [m for m in np.atleast_1d(d.get("missing", [""])).tolist() if m]
        print(f"{name}: life {d['life']:,.0f} scans, logged {d['days']:.0f} days, {len(d['ends'])} "
              f"days, {d['X'].shape[1]} input features, last-30-day share {np.mean(d['dtf'] <= 30):.2f}, "
              f"last-7-day share {np.mean(d['dtf'] <= 7):.2f}" + (f"; precursors not in this log: {', '.join(miss)}" if miss else ""))
    return tubes


def load_site_map(path):
    t = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    t.columns = [c.strip().lower() for c in t.columns]
    if not {"tube", "hospital"} <= set(t.columns):
        raise SystemExit("site map needs columns tube, hospital")
    key = lambda x: str(x).strip().replace("device_", "")
    return {key(r.tube): r.hospital.strip() for r in t.itertuples() if str(r.hospital).strip()}


def prior_pair(d, beta, eta):
    p7 = weibull_prior(d["age"], d["rate"], 7, beta, eta)
    return p7, np.maximum(weibull_prior(d["age"], d["rate"], 30, beta, eta), p7 + 1e-7)


def context(d, beta, eta, p7, p30, use_prior):
    a = d["age"] / eta
    if use_prior:
        return np.column_stack([a, a ** beta, np.log(d["rate"]), logit(p7), logit(p30)])
    return np.column_stack([a, np.log(d["rate"])])


def tensors(tubes, names, a, beta, eta, mu, sd, cmu, csd, use_prior, dev, tid, clip=None):
    out = []
    for n in names:
        d = tubes[n]
        Xn = (d["X"] - mu) / sd
        if clip is not None:
            Xn = np.clip(Xn, -clip, clip)
        idx = d["ends"][:, None] + np.arange(-a.window + 1, 1)[None, :]
        x = Xn[idx].astype(np.float32)
        p7, p30 = prior_pair(d, beta, eta)
        ctx = ((context(d, beta, eta, p7, p30, use_prior) - cmu) / csd).astype(np.float32)
        f = lambda v, dt=torch.float32: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)
        out.append({"x": f(x), "ctx": f(ctx), "y7": f(d["dtf"] <= 7), "y30": f(d["dtf"] <= 30),
                    "lp30": f(logit(p30)), "lr7": f(logit(p7 / p30)), "a_eta": f(d["age"] / eta),
                    "lrate": f(np.log(d["rate"])), "tube": f(np.full(len(d["ends"]), tid[n]), torch.long),
                    "feat_mean": x[:a.ref_windows, -1, :].mean(0), "p7_prior": p7, "p30_prior": p30})
    return out


def comm_cost(variant, fl, net, n_clients, rounds):
    """Bytes exchanged per training run (float32 parameters, both directions, all clients)."""
    P = sum(p.numel() for p in net.parameters())
    buf = sum(b.numel() for b in net.buffers())
    if fl == "central":
        return {"variant": variant, "params": P, "MB_per_round": 0.0, "MB_total": 0.0, "extra_once_bytes": 0}
    per = (P + buf) * 4 * 2
    if fl == "scaffold":
        per += P * 4 * 2                           # control variates both ways
    if fl == "fedbn":
        bnP = sum(t.numel() for k, t in net.state_dict().items() if k in bn_keys(net))
        per -= bnP * 4 * 2
    if fl == "cdp":
        per -= buf * 4 * 2                         # BN buffers are not shared
    extra = 100 * 2 * 8 if fl == "dtg" or "dtg2" in variant else 0   # the stage trend: 100 bins x (sum, count), once
    return {"variant": variant, "params": P, "MB_per_round": per * n_clients / 1e6,
            "MB_total": per * n_clients * rounds / 1e6, "extra_once_bytes": extra * n_clients}


def val_split(tubes, tests, seed, a):
    """Training tubes of a fold -> (all, validation, fitting); the validation tubes depend on the fold and the seed only,
    so every variant of a fold and seed uses the same ones."""
    names = [n for n in tubes if n not in tests]
    rng = np.random.default_rng(1000 * seed + sum(sorted(tubes).index(t) for t in tests))
    nval = min(a.val_tubes, max(1, len(names) // 5)) if a.val_tubes > 0 else 0
    val = list(rng.choice(names, size=nval, replace=False)) if nval else []
    return names, val, [n for n in names if n not in val]


def frame(tubes, tube, fold_name, p7, p30, **extra):
    d = tubes[tube]
    return pd.DataFrame({"tube": tube, "fold": fold_name, "dtf_days": d["dtf"], "age_scans": d["age"],
                         "scans_per_day": d["rate"], "y7": (d["dtf"] <= 7).astype(int), "y30": (d["dtf"] <= 30).astype(int),
                         "p7": p7, "p30": p30, **extra})


def run_fold(tubes, tests, fold_name, variant, seed, a, dev, logs):
    """-> (predictions for the test tube(s), predictions for the fold's validation tubes or None)."""
    fl, use_prior, extra = parse_variant(variant)
    names, val, fit = val_split(tubes, tests, seed, a)
    beta, eta = fit_weibull([tubes[n]["life"] for n in names])
    mu, sd, clip = input_scaling(tubes, fit, a)
    Cc = np.concatenate([context(tubes[n], beta, eta, *prior_pair(tubes[n], beta, eta), use_prior) for n in fit])
    cmu, csd = Cc.mean(0), Cc.std(0) + 1e-6
    tid = {n: i for i, n in enumerate(sorted(tubes))}
    args = (a, beta, eta, mu, sd, cmu, csd, use_prior, dev, tid, clip)
    clients = tensors(tubes, fit, *args)
    V = tensors(tubes, val, *args) if val else None
    cfg = RiskCfg(use_prior=use_prior, fl=fl, rounds=a.rounds, min_rounds=a.min_rounds, patience=a.patience,
                  local_epochs=a.local_epochs, batch=a.batch,
                  lr=a.lr, lam_mono=extra.get("lam_mono", a.lam_mono), lam_reg=extra.get("lam_reg", a.lam_reg), mu=a.mu, scaffold_lr=a.scaffold_lr,
                  cfl_warmup=a.cfl_warmup, cdp_z=extra.get("cdp_z", 1.0), cdp_clip=a.cdp_clip, device=dev, seed=seed)
    weights = None
    if extra.get("equal_w"):                         # control: every client gets the same aggregation weight
        weights = np.ones(len(fit))
    if fl == "dtg" or (fl == "fedbn" and "dtg2_tau" in extra):
        dp = None
        if "dp_eps" in extra:
            dp = {"eps": extra["dp_eps"], "r_clip": a.dp_rclip, "lag": a.dp_lag, "rng": np.random.default_rng(seed + 7)}
        weights, rho = dtg_weights_for([tubes[n] for n in fit], dp=dp)
        ns = np.array([len(tubes[n]["ends"]) for n in fit], float)
        extra_log = {}
        if "dtg2_tau" in extra:
            perm = None
            if extra.get("shuffle"):                 # control: the similarities are permuted across the clients
                ids = sorted(sorted(tubes).index(t) for t in tests)
                perm = np.random.default_rng([int(seed), 4242] + ids).permutation(len(fit))
            weights, s_k = dtg2_weights(rho, ns, extra["dtg2_tau"], perm=perm)
            extra_log = {"tau": extra["dtg2_tau"], "s_min": float(s_k.min()), "s_max": float(s_k.max()),
                         "shuffled": bool(extra.get("shuffle", False))}
        w_fedavg = ns / ns.sum()
        logs["dtg"].append({"variant": variant, "fold": fold_name, "seed": seed, "w_min": float(np.min(weights)),
                            "w_max": float(np.max(weights)), "w_uniform": 1.0 / len(fit),
                            "n_eff": float(1.0 / np.sum(np.asarray(weights) ** 2)),
                            "n_eff_fedavg": float(1.0 / np.sum(w_fedavg ** 2)), **extra_log,
                            **{f"w_{n}": float(w) for n, w in zip(fit, weights)}})
    t0 = time.time()
    model, curve, info = train(clients, V, cfg, clients[0]["x"].shape[2], clients[0]["ctx"].shape[1], weights)
    net = FailRiskNet(clients[0]["x"].shape[2], clients[0]["ctx"].shape[1]).to(dev)
    logs["curves"].append({"variant": variant, "fold": fold_name, "seed": seed, "curve": [float(c) for c in curve],
                           "best_round": int(np.nanargmin(curve)) + 1 if len(curve) and np.isfinite(np.nanmin(curve)) else None,
                           "seconds": time.time() - t0, **{k: v for k, v in info.items() if v is not None}})
    if not logs["comm"].get(variant):
        logs["comm"][variant] = comm_cost(variant, fl, net, len(fit), a.rounds)
    strip = lambda T: {k: v for k, v in T.items() if k not in ("feat_mean", "p7_prior", "p30_prior")}
    parts = []
    for test in tests:
        Tt = tensors(tubes, [test], *args)[0]
        p7, p30 = model.predict(net, strip(Tt), Tt["feat_mean"])
        parts.append(frame(tubes, test, fold_name, p7, p30, p7_prior=Tt["p7_prior"], p30_prior=Tt["p30_prior"]))
    vparts = []                                      # the selected model on the validation tubes (for choices made later)
    for vt, Tv in zip(val, V or []):
        p7, p30 = model.predict(net, strip(Tv), Tv["feat_mean"])
        vparts.append(frame(tubes, vt, fold_name, p7, p30))
    return pd.concat(parts, ignore_index=True), (pd.concat(vparts, ignore_index=True) if vparts else None)


def prior_fold(tubes, tests, fold_name):
    names = [n for n in tubes if n not in tests]
    beta, eta = fit_weibull([tubes[n]["life"] for n in names])
    parts = []
    for test in tests:
        d = tubes[test]
        p7, p30 = prior_pair(d, beta, eta)
        parts.append(pd.DataFrame({"tube": test, "fold": fold_name, "dtf_days": d["dtf"], "age_scans": d["age"],
                                   "scans_per_day": d["rate"], "y7": (d["dtf"] <= 7).astype(int),
                                   "y30": (d["dtf"] <= 30).astype(int), "p7": p7, "p30": p30,
                                   "beta": beta, "eta": eta}))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ parallel fold runs (--workers)
_W = {}


def _worker_init(tubes, a, dev, threads):
    torch.set_num_threads(max(1, int(threads)))
    _W.update(tubes=tubes, a=a, dev=dev)


def _worker_job(job):
    v, sd, fold, tests = job
    logs = {"dtg": [], "curves": [], "comm": {}}
    df, vdf = run_fold(_W["tubes"], tests, fold, v, sd, _W["a"], _W["dev"], logs)
    return v, sd, fold, df, vdf, logs


def save_fold(df, vdf, v, sd, fold, a, pred_fn):
    """Validation predictions first, then the test predictions (a complete test file means both are there)."""
    if vdf is not None:
        atomic_csv(vdf, os.path.join(a.out, "predval", f"{v}__seed{sd}__{fold}.csv"))
    atomic_csv(df, pred_fn(v, sd, fold))


def run_parallel(todo, tubes, a, dev, logs, lp, pred_fn):
    """Runs the fold jobs in --workers processes (spawn, also on Linux, because of CUDA). Every finished fold is
    written at once; folds that fail here are redone one after another by the caller. Results are the same as in
    a sequential run (each fold has its own seed)."""
    n, k, fails, t0 = len(todo), 0, 0, time.time()
    threads = max(1, (os.cpu_count() or 2) // a.workers)
    print(f"running {n} fold runs in {a.workers} parallel processes", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                 initargs=(tubes, a, dev, threads)) as ex:
            futs = {ex.submit(_worker_job, j): j for j in todo}
            for f in as_completed(futs):
                v, sd, fold, _ = futs[f]
                k += 1
                try:
                    _, _, _, df, vdf, lg = f.result()
                except Exception as e:
                    fails += 1
                    if fails <= 5:
                        print(f"  {v} seed {sd} fold {fold}: failed in a worker ({type(e).__name__}: {e}); redone later",
                              flush=True)
                    continue
                save_fold(df, vdf, v, sd, fold, a, pred_fn)
                logs["dtg"] += lg["dtg"]
                logs["curves"] += lg["curves"]
                for key, val in lg["comm"].items():
                    logs["comm"].setdefault(key, val)
                atomic_json(logs, lp)
                if k == n or k % max(1, n // 40) == 0:
                    el = time.time() - t0
                    print(f"  {k}/{n} fold runs done, {el / 60:.0f} min, about {el / k * (n - k) / 60:.0f} min left", flush=True)
    except Exception as e:
        print(f"parallel run stopped ({type(e).__name__}: {e}); the remaining folds run one after another", flush=True)
    if fails:
        print(f"{fails} fold runs failed in the workers; they are redone one after another", flush=True)


# ------------------------------------------------------------------ results made with other settings are never mixed
SIG_KEYS = ("input", "protocol", "seq_days", "min_dtf", "min_life", "window", "stride", "rounds", "min_rounds", "patience",
            "local_epochs", "batch", "lr", "val_tubes", "lam_mono", "lam_reg", "mu", "scaffold_lr", "cfl_warmup", "cdp_clip")
SIG_DEFAULTS = {"min_rounds": 0, "patience": 0, "format": 1}
FORMAT = 2                                           # 2: validation predictions are stored (predval/)


def _norm(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [_norm(y) for y in x]
    return x


def settings_signature(a, excluded):
    s = {k: _norm(getattr(a, k)) for k in SIG_KEYS}
    s["excluded"] = sorted(excluded)
    s["format"] = _norm(FORMAT)
    return s


def guard_settings(a, sig):
    """If --out holds results trained with other settings (e.g. another batch size or training length), they are
    moved to <out>/superseded_<time>/ so that --skip-existing never mixes them with the new ones."""
    fn = os.path.join(a.out, "config.json")
    if not os.path.exists(fn):
        return
    try:
        old = json.load(open(fn, encoding="utf-8"))
    except Exception:
        return
    old_sig = old.get("signature")
    if old_sig is None:                              # results made before the signature was stored
        args = old.get("args", {})
        old_sig = {k: _norm(args.get(k, SIG_DEFAULTS.get(k))) for k in SIG_KEYS}
        old_sig["excluded"] = sorted(old.get("excluded", {}))
    old_sig.setdefault("format", _norm(SIG_DEFAULTS["format"]))
    diff = [k for k in sig if _norm(old_sig.get(k)) != sig[k]]
    if not diff:
        return
    if getattr(a, "require_same_settings", False):
        raise SystemExit("settings differ from the results already in this folder (" +
                         ", ".join(f"{k} {old_sig.get(k)} -> {sig[k]}" for k in diff) + "); nothing was changed")
    dst = os.path.join(a.out, f"superseded_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(a.out):
        if name == "cache" or name.startswith("superseded_"):
            continue
        shutil.move(os.path.join(a.out, name), os.path.join(dst, name))
    os.makedirs(os.path.join(a.out, "pred"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "predval"), exist_ok=True)
    print("results in this folder were made with other settings (" +
          ", ".join(f"{k} {old_sig.get(k)} -> {sig[k]}" for k in diff) + f"); moved to {dst}, starting afresh", flush=True)


# ------------------------------------------------------------------ final-method candidates (no extra training)
SMOOTH_W = {7: (1, 3, 7), 30: (1, 3, 7, 14)}        # days of the causal moving average to choose from
FAMILIES = ("fedavg", "dtg2", "central",
            "dtg2-t0.05", "fedavg-eq", "dtg2-shuf")      # controls: 3-member ensembles of equal size
ONE_SEED = ("prior", "prior-s")


def seeds_of(v, seeds):
    return [0] if v in ONE_SEED else seeds


def smooth_days(df, w7, w30):
    """Causal moving average of each tube's daily probabilities over the last w days (w = 1: unchanged).
    Only earlier days of the same tube are used; P7 <= P30 is kept."""
    out = df.copy()
    out["p7"], out["p30"] = out["p7"].astype(float), out["p30"].astype(float)
    order = out.sort_values(["tube", "dtf_days"], ascending=[True, False]).index      # time order within each tube
    for c, w in (("p7", w7), ("p30", w30)):
        if w > 1:
            x = out.loc[order, c]
            out.loc[order, c] = x.groupby(out.loc[order, "tube"]).transform(
                lambda z: z.rolling(int(w), min_periods=1).mean()).to_numpy()
    out["p7"] = np.minimum(out["p7"], out["p30"])
    return out


def val_bce(df, H):
    """Validation loss as in training: binary cross-entropy per tube, averaged over tubes."""
    out = []
    for _, g in df.groupby("tube"):
        y = g[f"y{H}"].to_numpy(float)
        p = np.clip(g[f"p{H}"].to_numpy(float), 1e-6, 1 - 1e-6)
        out.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean(out)) if out else np.nan


def mean_of(frames, w=None):
    """Weighted average (default: plain) of the probabilities of prediction frames of the same days."""
    fr = [f.sort_values(["tube", "dtf_days"], ascending=[True, False]).reset_index(drop=True) for f in frames]
    for f in fr[1:]:
        if len(f) != len(fr[0]) or not np.array_equal(f.dtf_days.to_numpy(), fr[0].dtf_days.to_numpy()):
            raise ValueError("prediction frames of different days cannot be averaged")
    w = np.full(len(fr), 1.0 / len(fr)) if w is None else np.asarray(w, float) / np.sum(w)
    base = fr[0].copy()
    for c in ("p7", "p30"):
        base[c] = w @ np.stack([f[c].to_numpy(float) for f in fr])
    return base


def greedy_weights(vals, tol=1e-3):
    """Ensemble selection on the validation tubes (Caruana et al. 2004): start from the member with the lowest validation
    loss, then repeatedly add (with replacement) the member that lowers the validation loss of the average the most, as
    long as it does so by at least `tol` (relative). A single member is kept when no average is clearly better."""
    K = len(vals)
    fr = [f.sort_values(["tube", "dtf_days"], ascending=[True, False]).reset_index(drop=True) for f in vals]
    groups = [np.flatnonzero(fr[0].tube.to_numpy() == t) for t in fr[0].tube.unique()]
    Y = {H: fr[0][f"y{H}"].to_numpy(float) for H in (7, 30)}
    P = {H: np.stack([f[f"p{H}"].to_numpy(float) for f in fr]) for H in (7, 30)}

    def loss(c):                                     # = val_bce(7) + val_bce(30) of the weighted average
        out = 0.0
        for H in (7, 30):
            q = np.clip((c / c.sum()) @ P[H], 1e-6, 1 - 1e-6)
            el = -(Y[H] * np.log(q) + (1 - Y[H]) * np.log(1 - q))
            out += np.mean([el[g].mean() for g in groups])
        return out
    single = [loss(np.eye(K)[k]) for k in range(K)]
    c = np.eye(K)[int(np.argmin(single))]
    best = min(single)
    for _ in range(2 * K):
        cand = [loss(c + np.eye(K)[k]) for k in range(K)]
        k = int(np.argmin(cand))
        if not cand[k] < best * (1 - tol):
            break
        c, best = c + np.eye(K)[k], cand[k]
    return c / c.sum()


def posthoc(a, tubes, folds, seeds, preds, choices, tuned_req, report_order):
    """Final-method candidates built from the trained models; every choice is made per fold on that fold's validation
    tubes (all seeds), never on the test tube:
      <F>-ens   weighted average of the grid members of <F>-tuned (every penalty; DTG-v2: every tau as well), per fold
                and seed, weights by ensemble selection on the validation tubes (greedy_weights; it stays the single
                tuned member when no average is clearly better on validation)
      <x>-s     causal moving average of the daily probabilities over the last w days, w per fold and horizon from
                SMOOTH_W, used only when it lowers the validation loss by at least 0.1 %; applied alike to the prior,
                the data-only network, <F>-tuned and <F>-ens
    Returns the rows of posthoc_choices.csv."""
    def readv(v, sd, fold):
        fn = os.path.join(a.out, "predval", f"{v}__seed{sd}__{fold}.csv")
        return pd.read_csv(fn) if os.path.exists(fn) else None

    fams = [f for f in FAMILIES if f"{f}-tuned" in tuned_req]
    VAL, ens_rows = {}, []
    for f in fams:
        members = tune_members(f"{f}-tuned", a)
        for sd in seeds:
            parts = []
            for fold in folds:
                tests_m = [preds[(m, sd)][preds[(m, sd)].fold.astype(str) == str(fold)] for m in members]
                vs = [readv(m, sd, fold) for m in members]
                pick = next((c["chosen"] for c in choices if c["tuned"] == f"{f}-tuned" and str(c["fold"]) == str(fold)
                             and int(c["seed"]) == sd), None)
                ok = all(x is not None for x in vs)
                w = greedy_weights(vs) if ok else np.array([1.0 if m == pick else 0.0 for m in members])
                parts.append(mean_of(tests_m, w))
                VAL[(f"{f}-ens", sd, fold)] = mean_of(vs, w) if ok else None
                VAL[(f"{f}-tuned", sd, fold)] = readv(pick, sd, fold) if pick else None
                ens_rows.append({"variant": f"{f}-ens", "fold": fold, "seed": sd, "members_used": int((w > 0).sum()),
                                 **{f"weight_{m}": round(float(x), 4) for m, x in zip(members, w)}})
            preds[(f"{f}-ens", sd)] = pd.concat(parts, ignore_index=True)
    if ens_rows:
        pd.DataFrame(ens_rows).to_csv(os.path.join(a.out, "ensemble_weights.csv"), index=False)
    for sd in seeds:
        for fold, tests in folds.items():
            names, val, _ = val_split(tubes, tests, sd, a)
            if val:
                beta, eta = fit_weibull([tubes[n]["life"] for n in names])
                VAL[("prior", sd, fold)] = pd.concat([frame(tubes, n, fold, *prior_pair(tubes[n], beta, eta)) for n in val],
                                                     ignore_index=True)
            if ("nophys", sd) in preds:
                VAL[("nophys", sd, fold)] = readv("nophys", sd, fold)
    bases = ["prior"] + (["nophys"] if ("nophys", seeds[0]) in preds else []) + \
        [x for f in fams for x in (f"{f}-tuned", f"{f}-ens")]
    rows = []
    for x in bases:
        sds = seeds_of(x, seeds)
        out = {sd: [] for sd in sds}
        for fold in folds:
            w = {}
            for H in (7, 30):
                vs = [VAL.get((x, sd, fold)) for sd in seeds]
                if any(v is None for v in vs):
                    w[H] = 1                         # no validation predictions: no smoothing
                    continue
                loss = {k: np.mean([val_bce(smooth_days(v, k if H == 7 else 1, k if H == 30 else 1), H) for v in vs])
                        for k in SMOOTH_W[H]}
                k = min(loss, key=loss.get)          # smoothing only when it clearly helps on validation
                w[H] = k if loss[k] < loss[1] * (1 - 1e-3) else 1
            rows.append({"variant": f"{x}-s", "fold": fold, "w7_days": w[7], "w30_days": w[30]})
            for sd in sds:
                P = preds[(x, sd)]
                out[sd].append(smooth_days(P[P.fold.astype(str) == str(fold)], w[7], w[30]))
        for sd in sds:
            preds[(f"{x}-s", sd)] = pd.concat(out[sd], ignore_index=True)
    new = []                                         # report order: <F>-tuned, -tuned-s, -ens, -ens-s; prior, prior-s; ...
    for v in report_order:
        new.append(v)
        if v in bases:
            new.append(f"{v}-s")
        if v.endswith("-tuned") and v[:-len("-tuned")] in fams:
            new += [f"{v[:-len('-tuned')]}-ens", f"{v[:-len('-tuned')]}-ens-s"]
    report_order[:] = list(dict.fromkeys(new))
    return rows


def stability_analysis(tubes, a):
    """DTG weights without any training: (1) every log truncated to 30 / 50 / 70 % of its length, stage-aligned
    trend vs the index-based trend of the original DTG (equispaced over the whole log); (2) Laplace noise on the
    released trend statistics. Distances to the weights from the complete logs."""
    from scipy.stats import spearmanr
    names = sorted(tubes)
    full = [tubes[n] for n in names]
    D = eol_decline(full)
    w_stage, rho_stage = dtg_weights_for(full, D=D)
    w_index, _ = dtg_weights_index(full, D=D)
    ns = np.array([len(d["ends"]) for d in full], float)
    w2_full, _ = dtg2_weights(rho_stage, ns, DTG2_TAU)        # DTG-v2 (sample sizes of the full logs kept fixed)
    pd.DataFrame(rho_stage, index=names, columns=names).to_csv(os.path.join(a.out, "dtg_similarity_full.csv"))
    pd.DataFrame({"tube": names, "weight_stage": w_stage, "weight_index": w_index, "weight_dtg2": w2_full,
                  "weight_fedavg": ns / ns.sum()}).to_csv(os.path.join(a.out, "dtg_weights_full.csv"), index=False)
    rows = []
    for q in (0.3, 0.5, 0.7, 1.0):
        cut = []
        for d in full:
            k = max(int(len(d["hi"]) * q), 50)
            cut.append({**d, "hi": d["hi"][:k], "sc": d["sc"][:k]})
        for kind, fn, ref in (("stage-aligned", dtg_weights_for, w_stage),
                              ("index-based (original)", dtg_weights_index, w_index)):
            w, _ = fn(cut, D=D)
            rows.append({"analysis": "truncation", "setting": q, "trend": kind, "L1_to_full": float(np.abs(w - ref).sum()),
                         "spearman_to_full": float(spearmanr(w, ref)[0]), "w_spread": float(w.max() - w.min())})
        _, rho_cut = dtg_weights_for(cut, D=D)
        w2, _ = dtg2_weights(rho_cut, ns, DTG2_TAU)
        rows.append({"analysis": "truncation", "setting": q, "trend": f"DTG-v2 (tau {DTG2_TAU:g})",
                     "L1_to_full": float(np.abs(w2 - w2_full).sum()), "spearman_to_full": float(spearmanr(w2, w2_full)[0]),
                     "w_spread": float(w2.max() - w2.min())})
    w_ref, rho_ref = dtg_weights_for(full, D=D, dp={"eps": 1e12, "r_clip": a.dp_rclip, "lag": a.dp_lag,
                                                    "rng": np.random.default_rng(0)})
    w2_ref, _ = dtg2_weights(rho_ref, ns, DTG2_TAU)
    for eps in (0.1, 0.3, 1.0, 3.0, 10.0):
        L1, sp, L1b, spb = [], [], [], []
        for r in range(a.dp_draws):
            w, rho = dtg_weights_for(full, D=D, dp={"eps": eps, "r_clip": a.dp_rclip, "lag": a.dp_lag,
                                                    "rng": np.random.default_rng(1000 + r)})
            L1.append(np.abs(w - w_ref).sum())
            sp.append(spearmanr(w, w_ref)[0])
            w2, _ = dtg2_weights(rho, ns, DTG2_TAU)
            L1b.append(np.abs(w2 - w2_ref).sum())
            spb.append(spearmanr(w2, w2_ref)[0])
        rows.append({"analysis": "dp", "setting": eps, "trend": "stage-aligned", "L1_to_full": float(np.mean(L1)),
                     "spearman_to_full": float(np.nanmean(sp)), "w_spread": float("nan")})
        rows.append({"analysis": "dp", "setting": eps, "trend": f"DTG-v2 (tau {DTG2_TAU:g})", "L1_to_full": float(np.mean(L1b)),
                     "spearman_to_full": float(np.nanmean(spb)), "w_spread": float("nan")})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out, "dtg_stability.csv"), index=False)
    print(S.round(4).to_string(index=False))


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--src", default=FEATURES_DIR, help="folder containing tube_features.py (default: features/)")
    ap.add_argument("--pattern", default="device_*.csv")
    ap.add_argument("--out", default="results_failrisk")
    ap.add_argument("--cache", default="", help="feature cache folder (default <out>/cache); share it between runs")
    ap.add_argument("--protocol", default="loo", choices=["loo", "loho"])
    ap.add_argument("--input", default="daily-B2", choices=["daily-A2", "daily-B2"],
                    help="daily-A2: last --seq-days days of daily engineered features; daily-B2: daily-A2 + failure "
                         "precursors (see dtgexp/failrisk.py)")
    ap.add_argument("--exclude-file", default="", help="exclusion.csv from scripts/data_audit.py (tubes left out entirely)")
    ap.add_argument("--seq-days", type=int, default=60)
    ap.add_argument("--min-dtf", type=float, default=None,
                    help="drop samples whose prediction time is less than this many days before the failure "
                         "(their input contains the failure event itself); default 1")
    ap.add_argument("--full-log-frac", type=float, default=0.05,
                    help="robustness subset: tubes whose first logged scan count is at most this share of the life")
    ap.add_argument("--site-map", default="", help="tube -> hospital CSV (needed for --protocol loho)")
    ap.add_argument("--variants", default="prior,fedavg,dtg,central,nophys")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--window", type=int, default=50, help="(set to --seq-days for the daily inputs)")
    ap.add_argument("--stride", type=int, default=100, help="(not used by the daily inputs; kept in the run signature)")
    ap.add_argument("--min-life", type=float, default=5000, help="tubes with a shorter life (infant failures) are left out")
    ap.add_argument("--rounds", type=int, default=20, help="maximum number of rounds (centralized: epochs)")
    ap.add_argument("--min-rounds", type=int, default=0, help="early stopping: at least this many rounds")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: stop after this many rounds without improvement of the validation loss "
                         "(0 = always --rounds rounds); the round with the lowest validation loss is used either way")
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-tubes", type=int, default=3)
    ap.add_argument("--lam-mono", type=float, default=0.5)
    ap.add_argument("--lam-reg", type=float, default=0.01)
    ap.add_argument("--tune-reg", default="0.01,0.1,1",
                    help="lam_reg grid of the tuned variants (fedavg-tuned, dtg2-tuned, ...), chosen per fold on the validation tubes")
    ap.add_argument("--tune-tau", default="0.05",
                    help="DTG-v2 temperatures that dtg2-tuned chooses from together with the penalty (e.g. 0.02,0.05,0.1)")
    ap.add_argument("--mu", type=float, default=0.01, help="FedProx proximal weight")
    ap.add_argument("--scaffold-lr", type=float, default=0.05)
    ap.add_argument("--cfl-warmup", type=int, default=5)
    ap.add_argument("--cdp-clip", type=float, default=2.0)
    ap.add_argument("--dp-rclip", type=float, default=0.05, help="trend DP: clip of HI rate terms (HI per 1000 scans)")
    ap.add_argument("--dp-lag", type=int, default=2000, help="trend DP: records between the two ends of a rate term")
    ap.add_argument("--dp-draws", type=int, default=100)
    ap.add_argument("--alarm", default="7:0.06,30:0.25",
                    help="alarm thresholds per horizon (about twice the share of positive windows)")
    ap.add_argument("--ref-windows", type=int, default=100,
                    help="reference period (first windows of a tube) for CFL cluster assignment")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--analysis-only", action="store_true", help="only the DTG weight stability analysis (no training)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes for the fold runs (same results, faster; each needs its own GPU memory)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--require-same-settings", action="store_true",
                    help="stop (instead of moving the old results aside) when --out holds results made with other settings")
    a = ap.parse_args()
    if a.input.startswith("daily"):
        a.window = a.seq_days                     # the sequence branch reads the last seq_days days
    if a.min_dtf is None:
        a.min_dtf = 1.0 if a.input.startswith("daily") else 0.0
    os.makedirs(os.path.join(a.out, "pred"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "predval"), exist_ok=True)
    dev = resolve_device(a.device)
    print(f"device: {dev}; protocol: {a.protocol}")
    tubes = load_tubes(a)
    if len(tubes) < 4:
        raise SystemExit("need at least 4 tubes")
    widths = {n: d["X"].shape[1] for n, d in tubes.items()}
    if len(set(widths.values())) > 1:
        raise SystemExit(f"tubes have different numbers of input features: {widths}")
    if a.input in ("daily-A2", "daily-B2"):            # diagnostic only (scale from all tubes, no labels involved)
        mu_, sd_, c_ = input_scaling(tubes, list(tubes), a)
        R = np.concatenate([(d["X"][int(d.get("pad", 0)):] - mu_) / sd_ for d in tubes.values()])
        at = np.mean(np.abs(R) >= c_, 0) * 100
        feats = list(next(iter(tubes.values()))["features"])
        pre = [i for i, f in enumerate(feats) if f in PRECURSORS_V2]
        eng = [i for i in range(len(feats)) if i not in pre]
        msg = f"inputs at the clip limit (+-{c_:g} robust SD): engineered/workload {at[eng].mean():.1f}% on average"
        if pre:
            msg += f", precursors {at[pre].mean():.1f}% on average"
        print(msg + f"; highest: {feats[int(np.argmax(at))]} {at.max():.1f}%")
    if a.analysis_only:
        stability_analysis(tubes, a)
        return
    if a.protocol == "loho":
        if not a.site_map:
            raise SystemExit("--protocol loho needs --site-map")
        sm = load_site_map(a.site_map)
        hosp = {n: sm.get(n.replace("device_", ""), "") for n in tubes}
        missing = [n for n, h in hosp.items() if not h]
        if missing:
            raise SystemExit(f"tubes without a hospital in the site map: {missing}")
        folds = {h: sorted(n for n in tubes if hosp[n] == h) for h in sorted(set(hosp.values()))}
    else:
        folds = {n: [n] for n in sorted(tubes)}
    requested = [v.strip() for v in a.variants.split(",") if v.strip()]
    variants, report_order = [], []                 # variants to train; order of the report (members before tuned)
    for v in requested:
        if is_tuned(v):
            for m in tune_members(v, a):
                if m not in variants:
                    variants.append(m)
                if m not in report_order:
                    report_order.append(m)
        elif v not in variants:
            variants.append(v)
        if v not in report_order:
            report_order.append(v)
    for v in variants:
        if v != "prior":
            parse_variant(v)
    tuned_req = [v for v in requested if is_tuned(v)]
    seeds = [int(s) for s in a.seeds.split(",")]
    thr = {int(k): float(v) for k, v in (x.split(":") for x in a.alarm.split(","))}
    excluded = read_exclusions(a.exclude_file)
    sig = settings_signature(a, excluded)
    guard_settings(a, sig)
    json.dump({"args": vars(a), "signature": sig, "tubes": sorted(tubes), "excluded": excluded, "folds": folds,
               "device": dev, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__},
              open(os.path.join(a.out, "config.json"), "w", encoding="utf-8"), indent=1)
    if a.patience > 0:
        print(f"training length: early stopping on the validation tubes (at least {a.min_rounds}, at most {a.rounds} rounds, "
              f"stop after {a.patience} rounds without improvement; centralized: the same in epochs)")
    logs = {"dtg": [], "curves": [], "comm": {}}
    lp = os.path.join(a.out, "train_log.json")
    if a.skip_existing and os.path.exists(lp):
        try:
            old = json.load(open(lp, encoding="utf-8"))
            logs["dtg"], logs["curves"], logs["comm"] = old.get("dtg", []), old.get("curves", []), old.get("comm", {})
        except Exception:
            os.replace(lp, lp + ".damaged")
            print("train_log.json was damaged (power loss while writing): kept as train_log.json.damaged; "
                  "curves and DTG weights of folds finished before the interruption will be missing from the log")
    jobs = [(v, sd, fold) for v in variants for sd in ([0] if v == "prior" else seeds) for fold in folds]
    done = sum(1 for v, sd, fold in jobs if a.skip_existing and
               read_done(os.path.join(a.out, "pred", f"{v}__seed{sd}__{fold}.csv"), tubes, folds[fold]) is not None)
    print(f"fold runs: {len(jobs)} in total, {done} already complete, {len(jobs) - done} to do")
    pred_fn = lambda v, sd, fold: os.path.join(a.out, "pred", f"{v}__seed{sd}__{fold}.csv")
    if a.workers > 1:
        todo = [(v, sd, fold, folds[fold]) for v, sd, fold in jobs if v != "prior" and
                not (a.skip_existing and read_done(pred_fn(v, sd, fold), tubes, folds[fold]) is not None)]
        if todo:
            run_parallel(todo, tubes, a, dev, logs, lp, pred_fn)
        a.skip_existing = True                      # reuse what the workers wrote; failed folds are redone below
    preds = {}
    for v in variants:
        for sd in ([0] if v == "prior" else seeds):
            parts, t0 = [], time.time()
            for fold, tests in folds.items():
                fn = os.path.join(a.out, "pred", f"{v}__seed{sd}__{fold}.csv")
                if a.skip_existing and os.path.exists(fn):
                    df = read_done(fn, tubes, tests)
                    if df is not None:
                        parts.append(df)
                        continue
                    print(f"  {os.path.basename(fn)} is incomplete or was made with other settings - redoing this fold")
                if v == "prior":
                    df = prior_fold(tubes, tests, fold)
                    atomic_csv(df, fn)
                else:
                    df, vdf = run_fold(tubes, tests, fold, v, sd, a, dev, logs)
                    save_fold(df, vdf, v, sd, fold, a, pred_fn)
                parts.append(df)
                atomic_json(logs, lp)
            preds[(v, sd)] = pd.concat(parts, ignore_index=True)
            P = preds[(v, sd)]
            m30 = metrics(P.y30.to_numpy(), P.p30.to_numpy(), tube_weights(P.tube.to_numpy()))
            print(f"  {v:<10s} seed {sd}: AUC30 {m30['auc']:.3f}  ({time.time() - t0:.0f} s)")
    if tuned_req:                                   # nested selection: validation tubes only, never the test tube
        last = {}
        for c in logs["curves"]:
            cv = [x for x in c.get("curve", []) if x is not None and np.isfinite(x)]
            if cv:
                last[(c["variant"], str(c["fold"]), int(c["seed"]))] = float(min(cv))
        choices = []
        for tv in tuned_req:
            members = tune_members(tv, a)
            for sd in seeds:
                parts = []
                for fold, tests in folds.items():
                    vals = {m: last.get((m, str(fold), sd), np.nan) for m in members}
                    ok = {m: x for m, x in vals.items() if np.isfinite(x)}
                    pick = min(ok, key=ok.get) if ok else members[0]
                    src = preds[(pick, sd)]
                    df = src[src.fold.astype(str) == str(fold)].copy()
                    atomic_csv(df, os.path.join(a.out, "pred", f"{tv}__seed{sd}__{fold}.csv"))
                    parts.append(df)
                    choices.append({"tuned": tv, "fold": fold, "seed": sd, "chosen": pick,
                                    "selected_by": "validation loss" if ok else "fallback (no validation curve)",
                                    **{f"val_loss_{m}": vals[m] for m in members}})
                preds[(tv, sd)] = pd.concat(parts, ignore_index=True)
        C = pd.DataFrame(choices)
        C.to_csv(os.path.join(a.out, "tuning_choices.csv"), index=False)
        print("nested selection on the validation tubes (chosen configuration: number of fold-seed runs):")
        for tv, g in C.groupby("tuned"):
            print(f"  {tv}: " + ", ".join(f"{k} {n}" for k, n in g.chosen.value_counts().items()))
        if a.input.startswith("daily") and a.val_tubes > 0:
            ph = pd.DataFrame(posthoc(a, tubes, folds, seeds, preds, choices, tuned_req, report_order))
            ph.to_csv(os.path.join(a.out, "posthoc_choices.csv"), index=False)
            print("smoothing window chosen on the validation tubes (days; number of folds):")
            for v, g in ph.groupby("variant", sort=False):
                print(f"  {v}: 7 days " + ", ".join(f"w{k} {n}" for k, n in g.w7_days.value_counts().sort_index().items()) +
                      "; 30 days " + ", ".join(f"w{k} {n}" for k, n in g.w30_days.value_counts().sort_index().items()))
    variants = report_order
    if logs["dtg"]:
        pd.DataFrame(logs["dtg"]).to_csv(os.path.join(a.out, "dtg_weights.csv"), index=False)
    # training length actually used (rounds; centralized: epochs / local epochs), from the validation curves
    cur = {}
    for c in logs["curves"]:                         # the last entry per (variant, fold, seed) is the current one
        cv = [x for x in c.get("curve", []) if x is not None]
        if cv:
            cur[(c["variant"], str(c["fold"]), int(c["seed"]))] = c
    TL = []
    for (v, fold, sd), c in cur.items():
        if v not in variants and not any(v in tune_members(t, a) for t in tuned_req):
            continue
        cv = np.array([x if x is not None else np.nan for x in c["curve"]], float)
        per = a.local_epochs if parse_variant(v)[0] == "central" else 1
        run_r = float(c.get("rounds_run", len(cv) / per))
        best_r = (int(np.nanargmin(cv)) + 1) / per if np.isfinite(cv).any() else np.nan
        if a.patience > 0:                          # stopped by the maximum while still improving
            still = run_r >= a.rounds and best_r > a.rounds - a.patience
        else:                                        # fixed length: best round among the last two
            still = best_r >= run_r - 1
        TL.append({"variant": v, "fold": fold, "seed": sd, "rounds_run": run_r, "best_round": best_r,
                   "still_improving_at_end": bool(still)})
    rounds_mean = {}
    if TL:
        TLd = pd.DataFrame(TL)
        TLd.to_csv(os.path.join(a.out, "training_length.csv"), index=False)
        rounds_mean = TLd.groupby("variant").rounds_run.mean().to_dict()
    if logs["comm"]:
        Cm = pd.DataFrame(list(logs["comm"].values()))
        Cm["rounds_run_mean"] = [rounds_mean.get(v, a.rounds) for v in Cm.variant]
        Cm["MB_total"] = Cm.MB_per_round * Cm.rounds_run_mean
        logs["comm"] = {r["variant"]: r for r in Cm.to_dict("records")}
        Cm.to_csv(os.path.join(a.out, "communication.csv"), index=False)
    avg = {}
    for v in variants:
        runs = [preds[(v, s)].sort_values(["tube", "dtf_days"], ascending=[True, False]).reset_index(drop=True)
                for s in seeds_of(v, seeds)]                # the same day order for every variant (paired comparisons)
        base = runs[0][["tube", "fold", "dtf_days", "age_scans", "scans_per_day", "y7", "y30"]].copy()
        base["p7"] = np.mean([r.p7.to_numpy() for r in runs], 0)
        base["p30"] = np.mean([r.p30.to_numpy() for r in runs], 0)
        avg[v] = base
        base.to_csv(os.path.join(a.out, f"pred_mean__{v}.csv"), index=False, float_format="%.6g")
    report, rows = [], []
    say = lambda s="": (print(s), report.append(s))
    eps_of = {}
    for c in logs["curves"]:
        if "cdp_epsilon" in c:
            eps_of[c["variant"]] = c["cdp_epsilon"]
    for v in variants:
        for H in (7, 30):
            w = tube_weights(avg[v].tube.to_numpy())
            per_seed = [metrics(preds[(v, s)][f"y{H}"].to_numpy(), preds[(v, s)][f"p{H}"].to_numpy(),
                                tube_weights(preds[(v, s)].tube.to_numpy())) for s in seeds_of(v, seeds)]
            ma = metrics(avg[v][f"y{H}"].to_numpy(), avg[v][f"p{H}"].to_numpy(), w)
            lead, share, fa = alarm_stats(avg[v], f"p{H}", thr.get(H, 0.5), H)
            rows.append({"variant": v, "horizon_days": H, "auc": ma["auc"],
                         "auc_seed_sd": float(np.std([m["auc"] for m in per_seed])), "avg_precision": ma["ap"],
                         "brier": ma["brier"], "positive_share": float(np.average(avg[v][f"y{H}"], weights=w)),
                         "median_lead_days": lead, "tubes_alarmed_share": share, "false_alarm_time_pct": fa,
                         "alarm_threshold": thr.get(H, 0.5), "dp_epsilon": eps_of.get(v, np.nan)})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        say(f"protocol {a.protocol}; input {a.input}; tubes {len(tubes)}; folds {len(folds)}; "
            f"input features {next(iter(tubes.values()))['X'].shape[1]}; samples less than {a.min_dtf:g} day(s) "
            f"before failure excluded")
        say(S.round(3).to_string(index=False))
    tests = []
    member = lambda v: bool(re.search(r"-r[0-9.]+$", v) or re.fullmatch(r"dtg2-t[0-9.]+", v))  # fixed-setting grid member
    vs_fedavg = lambda v: (v not in ("prior", "fedavg", "nophys") and not member(v) and not is_tuned(v)
                           and not v.endswith(("-s", "-ens")))        # baselines, privacy and ablation variants
    pairs = [(v, "prior") for v in variants if v != "prior" and not member(v)] + \
            [(v, "fedavg") for v in variants if vs_fedavg(v)] + [("fedavg", "nophys")] + \
            [(v, "fedavg-tuned") for v in variants if v.startswith("dtg2") and is_tuned(v)] + \
            [("fedavg-tuned", "nophys"), ("fedavg-tuned", "central-tuned"), ("dtg2-tuned", "central-tuned")] + \
            [(f"{f}-ens-s", "prior-s") for f in FAMILIES] + [(f"{f}-tuned-s", "prior-s") for f in FAMILIES] + \
            [("dtg2-ens-s", "fedavg-ens-s"), ("fedavg-ens-s", "nophys-s"), ("fedavg-ens-s", "central-ens-s"),
             ("dtg2-ens-s", "central-ens-s"), ("prior-s", "prior")] + \
            [(v, "dtg2") for v in variants if v.startswith("dtg2-dp")]
    pairs = list(dict.fromkeys(pairs))
    for x, y in pairs:
        if x not in avg or y not in avg:
            continue
        m = avg[x][["tube", "y7", "y30"]].copy()
        for H in (7, 30):
            m[f"a{H}"], m[f"b{H}"] = avg[x][f"p{H}"].to_numpy(), avg[y][f"p{H}"].to_numpy()
            d, lo, hi, p_le0 = bootstrap_delta(m, f"a{H}", f"b{H}", f"y{H}", a.bootstrap)
            tests.append({"compare": f"{x} - {y}", "horizon_days": H, "delta_auc": d, "ci_low": lo, "ci_high": hi,
                          "bootstrap_share_delta_le_0": p_le0})
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    say("\nAUC differences, tube-level bootstrap (95% CI):")
    say(T.round(3).to_string(index=False))
    # robustness: only tubes whose logs start at installation (the log start is then not tied to the failure)
    full = sorted(n for n, d in tubes.items() if float(d["sc"][0]) <= a.full_log_frac * float(d["life"]))
    say(f"\nrobustness subset - logs starting at installation (first logged scan <= {a.full_log_frac:.0%} of life): "
        f"{len(full)} of {len(tubes)} tubes")
    rows_f, tests_f = [], []
    for v in variants:
        sub = avg[v][avg[v].tube.isin(full)]
        if sub.empty:
            continue
        for H in (7, 30):
            m = metrics(sub[f"y{H}"].to_numpy(), sub[f"p{H}"].to_numpy(), tube_weights(sub.tube.to_numpy()))
            rows_f.append({"variant": v, "horizon_days": H, "auc": m["auc"], "avg_precision": m["ap"], "brier": m["brier"],
                           "tubes": sub.tube.nunique()})
    for x, y in [(v, "prior") for v in variants if v != "prior" and not member(v)] + [("fedavg", "nophys")] + \
                [(v, "fedavg") for v in variants if vs_fedavg(v)] + \
                [("dtg2-tuned", "fedavg-tuned"), ("fedavg-tuned", "nophys"), ("fedavg-ens-s", "prior-s"),
                 ("dtg2-ens-s", "prior-s"), ("dtg2-ens-s", "fedavg-ens-s")]:
        if x not in avg or y not in avg:
            continue
        m = avg[x][["tube", "y7", "y30"]].copy()
        for H in (7, 30):
            m[f"a{H}"], m[f"b{H}"] = avg[x][f"p{H}"].to_numpy(), avg[y][f"p{H}"].to_numpy()
        m = m[m.tube.isin(full)]
        if m.tube.nunique() < 3:
            continue
        for H in (7, 30):
            d, lo, hi, p_le0 = bootstrap_delta(m, f"a{H}", f"b{H}", f"y{H}", a.bootstrap)
            tests_f.append({"compare": f"{x} - {y}", "horizon_days": H, "delta_auc": d, "ci_low": lo, "ci_high": hi,
                            "bootstrap_share_delta_le_0": p_le0})
    if rows_f:
        pd.DataFrame(rows_f).to_csv(os.path.join(a.out, "summary_fulllog.csv"), index=False)
        say(pd.DataFrame(rows_f).round(3).to_string(index=False))
    if tests_f:
        pd.DataFrame(tests_f).to_csv(os.path.join(a.out, "tests_fulllog.csv"), index=False)
        say(pd.DataFrame(tests_f).round(3).to_string(index=False))
    if TL:
        g = TLd.groupby("variant").agg(runs=("rounds_run", "size"), rounds_run_median=("rounds_run", "median"),
                                       best_round_median=("best_round", "median"),
                                       still_improving_at_end=("still_improving_at_end", "mean"))
        g = g.reindex([v for v in variants if v in g.index])
        say("\ntraining length (rounds; centralized: epochs) and the round with the lowest validation loss; "
            "'still improving at end' = share of runs stopped by the maximum while the validation loss was still falling:")
        say(g.round(2).to_string())
    if logs["comm"]:
        say("\ncommunication per training run (all clients, both directions):")
        say(pd.DataFrame(list(logs["comm"].values())).round(3).to_string(index=False))
    with open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    say(f"\nwritten to {a.out}: summary.csv, tests.csv, report.txt, train_log.json (curves, DTG weights, communication), "
        f"pred_mean__<variant>.csv, pred/<variant>__seed<s>__<fold>.csv")


if __name__ == "__main__":
    main()
