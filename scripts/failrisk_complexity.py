#!/usr/bin/env python3
"""Computational and communication complexity of the failure-risk network, from a finished run (no training).

  python scripts/failrisk_complexity.py --run results/B2/loo --cache results/cache --out results/comparison

Model: parameters, size, multiply-accumulates per prediction (one tube-day), inference time (CPU, and GPU when
present). Training: median time per fold run and per round from train_log.json (hardware of the run). Communication:
megabytes per round (all clients, both directions), rounds actually run (early stopping) and the total, from
communication.csv; DTG adds a one-off upload of its trend (100 bins x sum and count per client).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtgexp.cli import resolve_device, setup_console  # noqa: E402
from dtgexp.failrisk import FailRiskNet  # noqa: E402

CACHE_TAG = {"daily-A2": "daily2A", "daily-B2": "daily2B", "daily-A": "dailyA", "daily-B": "dailyB2"}


def macs(net, C, E, L):
    total = [0]

    def conv(m, i, o):
        total[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin(m, i, o):
        total[0] += o.numel() * m.in_features
    hooks = [m.register_forward_hook(conv if isinstance(m, nn.Conv1d) else lin)
             for m in net.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, L, C), torch.zeros(1, E))
    for h in hooks:
        h.remove()
    return int(total[0])


def latency_ms(net, C, E, L, batch, dev, reps=30):
    net = net.to(dev).eval()
    x, c = torch.zeros(batch, L, C, device=dev), torch.zeros(batch, E, device=dev)
    with torch.no_grad():
        for _ in range(5):
            net(x, c)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            net(x, c)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a finished run folder (loo) of scripts/failrisk.py")
    ap.add_argument("--cache", required=True, help="the daily feature cache of that run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(a.run, "config.json"), encoding="utf-8"))["args"]
    L = int(cfg.get("seq_days", 60))
    hits = sorted(glob.glob(os.path.join(a.cache, f"*__{CACHE_TAG.get(cfg['input'], '')}_L{L}.npz")))
    if not hits:
        raise SystemExit(f"no cached features of input {cfg['input']} in {a.cache}")
    C = len(np.load(hits[0], allow_pickle=False)["features"])
    dev = resolve_device(a.device)
    rows = []
    for name, E in (("proposed network (with the physics prior)", 5), ("data-only network", 2)):
        net = FailRiskNet(C, E)
        P = sum(p.numel() for p in net.parameters())
        r = {"model": name, "input_features": C, "input_days": L, "parameters": P,
             "size_MB_float32": round(P * 4 / 1e6, 3), "MACs_per_prediction": macs(net, C, E, L),
             "cpu_ms_one_prediction": round(latency_ms(net, C, E, L, 1, "cpu"), 3),
             "cpu_ms_per_1000_predictions": round(latency_ms(net, C, E, L, 1000, "cpu"), 2)}
        if dev.startswith("cuda"):
            r["gpu_ms_per_1000_predictions"] = round(latency_ms(net, C, E, L, 1000, dev), 3)
        rows.append(r)
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(a.out, "table_complexity_model.csv"), index=False)
    print(M.to_string(index=False))
    tr = []
    lg = os.path.join(a.run, "train_log.json")
    if os.path.exists(lg):
        for c in json.load(open(lg, encoding="utf-8")).get("curves", []):
            if c.get("seconds") and c.get("curve"):
                n = float(c.get("rounds_run", len(c["curve"])))
                tr.append({"variant": c["variant"], "seconds": float(c["seconds"]), "rounds": n})
    T = pd.DataFrame(tr)
    if not T.empty:
        T = T.groupby("variant").agg(fold_runs=("seconds", "size"), median_seconds_per_run=("seconds", "median"),
                                     median_rounds=("rounds", "median")).reset_index()
        T["median_seconds_per_round"] = T.median_seconds_per_run / T.median_rounds
    cm = os.path.join(a.run, "communication.csv")
    if os.path.exists(cm):
        Cm = pd.read_csv(cm)
        keep = [c for c in ("variant", "MB_per_round", "rounds_run_mean", "MB_total", "extra_once_bytes") if c in Cm.columns]
        T = Cm[keep] if T.empty else T.merge(Cm[keep], on="variant", how="outer")
    if not T.empty:
        T = T.round(3)
        T.to_csv(os.path.join(a.out, "table_complexity_training.csv"), index=False)
        print(T.to_string(index=False))
    print(f"written to {a.out}: table_complexity_model.csv, table_complexity_training.csv "
          f"(training times are those of the machine that ran {a.run})")


if __name__ == "__main__":
    main()
