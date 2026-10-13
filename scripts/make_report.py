#!/usr/bin/env python3
"""Comparison report: every table of the paper from the run folders (no training).

  python scripts/make_report.py --root results

Folders under --root (written by reproduce.py): audit/ (data_audit.csv, exclusion.csv), A2/ and B2/, each with loo/
(leave one tube out) and loho/ (leave one hospital out); B2 is the primary input design, A2 leaves out the failure
precursors. When the audit excluded tubes, A2_incl/ and B2_incl/ hold the models retrained without them.

AUC differences use the tube-weighted AUC of all tubes as the point estimate and the 2.5th and 97.5th percentiles of
a tube-level bootstrap as the 95% CI (MET = CI above 0; "small" = improvement below 0.02). Samples less than 1 day
before failure are left out. The main comparisons are reported for the FINAL MODEL (ensemble selection over the
tuning grid, then a causal moving average of the daily probabilities with a window chosen per fold on the validation
tubes; every method, the prior included, smoothed the same way) and for the SINGLE TUNED MODEL (the grid member with
the lowest validation loss, no smoothing). The test tube never takes part in any choice.
Further sections: DTG-v2 by temperature, federated baselines, controls for the DTG-v2 gain (equal client weights,
permuted similarities), effective client weights, privacy, physics ablation and physical consistency, DTG weights
from truncated logs and under DP, complexity, training length and nested-selection choices.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dtgexp.failrisk import bootstrap_delta, metrics, tube_weights  # noqa: E402

MAIN = ["prior", "prior-s", "nophys", "nophys-s", "fedavg-ens-s", "dtg2-ens-s", "central-ens-s", "fedavg-ens", "dtg2-ens",
        "central-ens", "fedavg-tuned", "dtg2-tuned", "central-tuned", "fedavg-tuned-s", "dtg2-tuned-s", "central-tuned-s",
        "fedavg", "dtg", "dtg2", "central"]
FIXED_LAB = {"prior": "Physics prior", "nophys": "Data-only network", "fedavg-tuned": "Proposed (FedAvg, tuned)",
             "dtg2-tuned": "Proposed (DTG-v2, tuned)", "central-tuned": "Proposed (centralized, tuned)",
             "dtg": "DTG, original weighting (fixed)"}


def label(v):
    """Readable name of a variant (also of grid members such as dtg2-t0.02-r0.1 and final models such as fedavg-ens-s)."""
    if v.endswith("-s"):
        return label(v[:-len("-s")]) + " + smoothing"
    if v.endswith("-ens"):
        b = v[:-len("-ens")]
        name = {"fedavg": "FedAvg", "dtg2": "DTG-v2", "central": "centralized", "dtg2-t0.05": "DTG-v2, tau 0.05",
                "fedavg-eq": "FedAvg, equal client weights", "dtg2-shuf": "DTG-v2, shuffled similarities"}.get(b, b)
        return f"Proposed ({name}, ensemble)"
    m = re.fullmatch(r"(fedavg-eq|dtg2-shuf)(?:-r([0-9.]+))?", v)       # controls
    if m:
        name = "FedAvg, equal client weights" if m.group(1) == "fedavg-eq" else "DTG-v2, shuffled similarities (tau 0.05)"
        return f"{name}, penalty {m.group(2) or '0.01'} (fixed)"
    if v in ("fedavg-eq-tuned", "dtg2-shuf-tuned"):
        return ("FedAvg, equal client weights" if v.startswith("fedavg") else "DTG-v2, shuffled similarities") + \
            " (penalty tuned)"
    if v in FIXED_LAB:
        return FIXED_LAB[v]
    m = re.fullmatch(r"(.+)-m([0-9.]+)", v)
    if m:
        return label(m.group(1)) + (", without the monotonicity loss" if float(m.group(2)) == 0
                                    else f", monotonicity weight {m.group(2)}")
    m = re.fullmatch(r"(dtg2?)-dp([0-9.]+)", v)
    if m:
        return f"{'DTG-v2' if m.group(1) == 'dtg2' else 'DTG'}, trend released under DP (epsilon {m.group(2)})"
    m = re.fullmatch(r"cdp([0-9.]+)", v)
    if m:
        if float(m.group(1)) == 0:
            return "DP-FedAvg setting without noise (equal weights, clipping, local BN statistics)"
        return f"Client-level DP-FedAvg (noise multiplier {m.group(1)})"
    m = re.fullmatch(r"fedbn-dtg2(?:-t([0-9.]+))?", v)
    if m:
        return f"FedBN with DTG-v2 weights (tau {m.group(1) or '0.05'}), penalty 0.01 (fixed)"
    if v in ("fedprox", "scaffold", "fedbn", "cfl"):
        return {"fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedbn": "FedBN", "cfl": "CFL"}[v] + ", penalty 0.01 (fixed)"
    m = re.fullmatch(r"dtg2-t([0-9.]+)-tuned", v)
    if m:
        return f"DTG-v2, tau {m.group(1)} (penalty tuned)"
    m = re.fullmatch(r"(fedavg|central|dtg2(?:-t[0-9.]+)?)(?:-r([0-9.]+))?", v)
    if m:
        b, r = m.group(1), m.group(2) or "0.01"
        name = {"fedavg": "FedAvg", "central": "Centralized"}.get(b) or \
            ("DTG-v2, tau " + (b.split("-t")[1] if "-t" in b else "0.05"))
        return f"{name}, penalty {r} (fixed)"
    return v


def tau_of(v):
    m = re.fullmatch(r"dtg2-t([0-9.]+)-tuned", v)
    return float(m.group(1)) if m else np.nan


def variants_in(folder):
    """Variants with predictions in a run folder: the main ones first, then tau-specific tuned ones, then the rest."""
    import glob
    have = [os.path.basename(f)[len("pred_mean__"):-4] for f in glob.glob(os.path.join(folder, "pred_mean__*.csv"))]
    first = [v for v in MAIN if v in have]
    taus = sorted((v for v in have if np.isfinite(tau_of(v))), key=tau_of)
    return first + taus + sorted(v for v in have if v not in first and v not in taus)


FA, DT = "fedavg-tuned", "dtg2-tuned"
MIN_DTF = 1.0


def read(p):
    return pd.read_csv(p) if p and os.path.exists(p) else None


def preds(folder, v):
    P = read(os.path.join(folder, f"pred_mean__{v}.csv")) if folder else None
    return None if P is None else P[P.dtf_days >= MIN_DTF].reset_index(drop=True)


def auc_brier(P, H, keep=None):
    if P is None:
        return np.nan, np.nan
    if keep is not None:
        P = P[P.tube.isin(keep)]
    if P.tube.nunique() < 3:
        return np.nan, np.nan
    m = metrics(P[f"y{H}"].to_numpy(), P[f"p{H}"].to_numpy(), tube_weights(P.tube.to_numpy()))
    return m["auc"], m["brier"]


def paired(Pa, Pb, H, keep=None, reps=2000):
    """AUC(Pa) - AUC(Pb) on the same days, tube-level bootstrap."""
    if Pa is None or Pb is None:
        return None
    m = Pa[["tube", "dtf_days", f"y{H}"]].copy()
    m["a"] = Pa[f"p{H}"].to_numpy()
    k = Pb.set_index(["tube", "dtf_days"])[f"p{H}"]
    m["b"] = k.reindex(list(zip(m.tube, m.dtf_days))).to_numpy()
    m = m.dropna(subset=["b"])
    if keep is not None:
        m = m[m.tube.isin(keep)]
    if m.tube.nunique() < 3:
        return None
    return bootstrap_delta(m.rename(columns={f"y{H}": "y"}), "a", "b", "y", reps)


def mono_violation_by_tube(P, H=30, gap_days=30, max_dlog_rate=0.3, rel=0.05, pairs=20000, seed=0):
    """Physical consistency with wear-out (Weibull shape > 1: the hazard rises with age). The quantity checked is the
    workload-normalised cumulative hazard -ln(1 - P_H) / (scans per day), which for the physics prior increases with
    age whatever the workload. Per tube: share of day pairs, at least `gap_days` apart and with a similar workload
    (|change of log scans per day| < max_dlog_rate), in which it is more than `rel` (relative) lower on the later day.
    Random pairs with a fixed seed; two models with predictions on the same days are checked on the same pairs."""
    out = {}
    if P is None:
        return out
    rng = np.random.default_rng(seed)
    for t, g in P.groupby("tube"):
        g = g.sort_values("dtf_days", ascending=False)
        n = len(g)
        if n < gap_days + 2:
            continue
        i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
        e, l_ = np.minimum(i, j), np.maximum(i, j)
        day = -g.dtf_days.to_numpy(float)
        rate = np.maximum(g.scans_per_day.to_numpy(float), 1.0)
        lr = np.log(rate)
        q = -np.log1p(-np.clip(g[f"p{H}"].to_numpy(float), 0.0, 1 - 1e-9)) / rate
        ok = (day[l_] - day[e] >= gap_days) & (np.abs(lr[l_] - lr[e]) < max_dlog_rate)
        if ok.sum() >= 50:
            out[t] = float(np.mean(q[l_][ok] < (1 - rel) * q[e][ok]))
    return out


def mono_violation(P, H=30, **kw):
    """Share of day pairs with a falling hazard (mono_violation_by_tube), averaged over tubes."""
    d = mono_violation_by_tube(P, H, **kw)
    return float(np.mean(list(d.values()))) if d else np.nan


def mono_delta(Pa, Pb, H, reps=2000, seed=1):
    """Share of day pairs with a falling hazard, model a minus model b, on the same pairs; tube-level bootstrap.
    -> (share a, share b, difference, 2.5th and 97.5th percentile, tubes) or None."""
    a, b = mono_violation_by_tube(Pa, H), mono_violation_by_tube(Pb, H)
    ts = sorted(set(a) & set(b))
    if len(ts) < 3:
        return None
    d = np.array([a[t] - b[t] for t in ts])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    return (float(np.mean([a[t] for t in ts])), float(np.mean([b[t] for t in ts])), float(d.mean()),
            float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(ts))


def verdict(res):
    if res is None:
        return "not run", np.nan, np.nan, np.nan
    d, lo, hi, _ = res
    if lo <= 0:
        return "not met", d, lo, hi
    return ("MET" if d >= 0.02 else "MET, small (<0.02)"), d, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--full-log-frac", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "comparison")
    os.makedirs(out, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    E = read(os.path.join(a.root, "audit", "exclusion.csv"))
    excluded = set(E[E.excluded.astype(str).str.lower().isin(["true", "1"])].tube) if E is not None else set()
    D = read(os.path.join(a.root, "audit", "data_audit.csv"))
    start = {} if D is None else dict(zip(D.tube, D.log_start_share_of_life))
    incl_root = "_incl" if excluded and os.path.isdir(os.path.join(a.root, "B2_incl")) else ""
    run = lambda k, proto="loo": os.path.join(a.root, f"{k}{incl_root}", proto)
    run_all = lambda k, proto="loo": os.path.join(a.root, k, proto)
    tubes_incl = None
    P0 = preds(run("B2"), "prior")
    if P0 is not None:
        tubes_incl = sorted(set(P0.tube))
    full = [t for t in (tubes_incl or []) if start.get(t, 1.0) <= a.full_log_frac]
    say(f"tubes excluded by the data-quality criteria: {len(excluded)}" + (f" ({', '.join(sorted(excluded))})" if excluded else ""))
    if E is not None:
        for r in E.itertuples():
            if str(r.excluded).lower() in ("true", "1"):
                say(f"  {r.tube}: {r.reasons}")
    say(f"tubes in the primary analysis: {len(tubes_incl or [])}; of these, logs starting at installation: {len(full)}")

    # 1 overview, all inputs, models trained on all tubes, evaluated on the tubes of the primary analysis
    def bt(folder):
        """', batch 32, early stopping 20-100 rounds' etc. from the run's config.json."""
        try:
            args = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))["args"]
            s = f", batch {int(args['batch'])}"
            if int(args.get("patience", 0) or 0) > 0:
                return s + f", early stopping {int(args.get('min_rounds', 0) or 0)}-{int(args['rounds'])} rounds"
            return s + f", {int(args['rounds'])} rounds"
        except Exception:
            return ""
    designs = [(f"daily A2 (engineered features{bt(run_all('A2'))})", run_all("A2")),
               (f"daily B2 (with failure precursors{bt(run_all('B2'))})", run_all("B2"))]
    if incl_root:
        designs += [("daily A2, retrained without excluded tubes", run("A2")),
                     ("daily B2, retrained without excluded tubes", run("B2"))]
    rows = []
    for name, folder in designs:
        if not folder or not os.path.isdir(folder):
            continue
        if any(r["input"] == name for r in rows):     # two designs with the same label: keep them apart
            name += f" [{os.path.basename(os.path.dirname(os.path.dirname(folder))) or folder}]"
        for v in variants_in(folder):
            P = preds(folder, v)
            if P is None:
                continue
            a7, _ = auc_brier(P, 7, tubes_incl)
            a30, b30 = auc_brier(P, 30, tubes_incl)
            f7, _ = auc_brier(P, 7, full) if full else (np.nan, np.nan)
            rows.append({"input": name, "variant": v, "method": label(v), "auc_7d": a7, "auc_30d": a30, "brier_30d": b30,
                         "auc_7d_install_start": f7})
    O = pd.DataFrame(rows)
    O.round(3).to_csv(os.path.join(out, "table_inputs_overview.csv"), index=False)
    if not O.empty:                                  # the report lists the main methods; the csv has every grid member
        Os = O[O.variant.isin(MAIN) | O.variant.map(lambda v: np.isfinite(tau_of(v)))].drop(columns="variant")
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            say("\nAUC by input design (leave one tube out; evaluated on the tubes of the primary analysis; samples < 1 day "
                "before failure left out; every fixed-setting grid member: table_inputs_overview.csv)")
            say(Os.round(3).to_string(index=False))

    # 2 main comparisons, for the final model (ensemble + smoothing) and for the single tuned model
    B, A = run("B2"), run("A2")
    Bh = run("B2", "loho") if os.path.isdir(run("B2", "loho")) else run_all("B2", "loho")
    p = lambda folder, v: preds(folder, v)
    reps = a.bootstrap

    def block(fa, dt, pr, nph, ce, tag):
        F, D = label(fa), label(dt)
        c = [(f"PRIMARY  B2: {F} vs {label(pr)}, 7 days", paired(p(B, fa), p(B, pr), 7, None, reps)),
             ("  check: logs starting at installation", paired(p(B, fa), p(B, pr), 7, full, reps) if full else None),
             ("  check: all tubes (no exclusions)", paired(p(run_all("B2"), fa), p(run_all("B2"), pr), 7, None, reps)),
             (f"KEY SECONDARY B2: {D} vs prior, 7 days", paired(p(B, dt), p(B, pr), 7, None, reps)),
             ("  check: logs starting at installation", paired(p(B, dt), p(B, pr), 7, full, reps) if full else None),
             (f"KEY SECONDARY B2: {D} vs prior, 30 days", paired(p(B, dt), p(B, pr), 30, None, reps)),
             ("  check: logs starting at installation", paired(p(B, dt), p(B, pr), 30, full, reps) if full else None),
             (f"KEY SECONDARY B2: {D} vs {F}, 7 days", paired(p(B, dt), p(B, fa), 7, None, reps)),
             ("  check: logs starting at installation", paired(p(B, dt), p(B, fa), 7, full, reps) if full else None),
             (f"KEY SECONDARY B2: {D} vs {F}, 30 days", paired(p(B, dt), p(B, fa), 30, None, reps)),
             ("  check: logs starting at installation", paired(p(B, dt), p(B, fa), 30, full, reps) if full else None),
             (f"secondary B2: {F} vs prior, 30 days", paired(p(B, fa), p(B, pr), 30, None, reps)),
             ("  check: logs starting at installation", paired(p(B, fa), p(B, pr), 30, full, reps) if full else None),
             (f"secondary B2, leave one hospital out: {F} vs prior, 7 days", paired(p(Bh, fa), p(Bh, pr), 7, None, reps)),
             (f"secondary B2, leave one hospital out: {F} vs prior, 30 days", paired(p(Bh, fa), p(Bh, pr), 30, None, reps)),
             (f"secondary B2, leave one hospital out: {D} vs prior, 7 days", paired(p(Bh, dt), p(Bh, pr), 7, None, reps)),
             (f"secondary B2, leave one hospital out: {D} vs prior, 30 days", paired(p(Bh, dt), p(Bh, pr), 30, None, reps)),
             (f"secondary B2, leave one hospital out: {D} vs {F}, 7 days", paired(p(Bh, dt), p(Bh, fa), 7, None, reps)),
             (f"secondary B2, leave one hospital out: {D} vs {F}, 30 days", paired(p(Bh, dt), p(Bh, fa), 30, None, reps)),
             (f"secondary A2: {F} vs prior, 7 days", paired(p(A, fa), p(A, pr), 7, None, reps)),
             (f"secondary A2: {F} vs prior, 30 days", paired(p(A, fa), p(A, pr), 30, None, reps)),
             (f"secondary precursors: B2 vs A2, {F}, 7 days", paired(p(B, fa), p(A, fa), 7, None, reps)),
             (f"secondary precursors: B2 vs A2, {D}, 7 days", paired(p(B, dt), p(A, dt), 7, None, reps)),
             (f"physics value, B2: {F} vs {label(nph)}, 7 days", paired(p(B, fa), p(B, nph), 7, None, reps)),
             (f"physics value, B2: {F} vs {label(nph)}, 30 days", paired(p(B, fa), p(B, nph), 30, None, reps)),
             (f"reference B2: {label(ce)} vs prior, 7 days", paired(p(B, ce), p(B, pr), 7, None, reps)),
             (f"reference B2: {F} vs {label(ce)}, 7 days", paired(p(B, fa), p(B, ce), 7, None, reps)),
             (f"reference B2: {F} vs {label(ce)}, 30 days", paired(p(B, fa), p(B, ce), 30, None, reps)),
             (f"reference B2: {D} vs {label(ce)}, 7 days", paired(p(B, dt), p(B, ce), 7, None, reps)),
             (f"reference B2: {D} vs {label(ce)}, 30 days", paired(p(B, dt), p(B, ce), 30, None, reps))]
        rows = []
        for name, res in c:
            r, d, lo, hi = verdict(res)
            if name.startswith("reference") and res is not None:
                r = "(reference)"
            rows.append({"model": tag, "comparison": name.strip(), "result": r, "delta_auc": d, "ci_low": lo, "ci_high": hi})
            say(f"  {name:<92s} {r:<19s}" + (f" delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(d) else ""))
        return rows

    V = []
    if p(B, "fedavg-ens-s") is not None:
        say("\nFINAL MODEL: ensemble of the tuning grid + causal smoothing, both chosen on the validation tubes; every method, "
            "the prior included, smoothed the same way (MET = 95% CI of the AUC difference above 0; 'small' = below 0.02):")
        V += block("fedavg-ens-s", "dtg2-ens-s", "prior-s", "nophys-s", "central-ens-s", "final (ensemble + smoothing)")
    say("\nSINGLE TUNED MODEL (setting chosen on the validation tubes, no ensemble, no smoothing):")
    V += block(FA, DT, "prior", "nophys", "central-tuned" if p(B, "central-tuned") is not None else "central",
               "single tuned model")
    pd.DataFrame(V).round(4).to_csv(os.path.join(out, "criteria.csv"), index=False)

    # 2a smoothing windows chosen on the validation tubes
    ph = read(os.path.join(B, "posthoc_choices.csv"))
    if ph is not None and not ph.empty:
        g = ph.groupby("variant", sort=False).agg(folds=("fold", "size"), w7_median=("w7_days", "median"),
                                                  w30_median=("w30_days", "median")).reset_index()
        g.insert(1, "method", g.variant.map(label))
        g.to_csv(os.path.join(out, "table_smoothing.csv"), index=False)
        say("\nSmoothing window in B2 (days of the causal moving average, chosen per fold on the validation tubes):")
        say(g.to_string(index=False))

    # 2b' DTG-v2 temperature: the penalty tuned on the validation tubes at each tau (reported, not judged)
    tv = [v for v in variants_in(B) if np.isfinite(tau_of(v))] if B and os.path.isdir(B) else []
    if tv:
        try:
            grid = json.load(open(os.path.join(B, "config.json"), encoding="utf-8"))["args"].get("tune_tau", "0.05")
        except Exception:
            grid = "?"
        rows_t = []
        for v in tv:
            r = {"tau": tau_of(v)}
            for H in (7, 30):
                r[f"auc_{H}d"] = auc_brier(p(B, v), H)[0]
                for other, key in (("prior", "vs_prior"), (FA, "vs_fedavg_tuned")):
                    res = paired(p(B, v), p(B, other), H, None, reps)
                    d, lo, hi = (res[0], res[1], res[2]) if res else (np.nan, np.nan, np.nan)
                    r[f"{key}_{H}d"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(d) else ""
            rows_t.append(r)
        Tt = pd.DataFrame(rows_t)
        Tt.round(3).to_csv(os.path.join(out, "table_dtg2_tau.csv"), index=False)
        say(f"\nDTG-v2 by temperature tau in B2 (penalty chosen on the validation tubes at each tau; DTG-v2 (tuned) chooses "
            f"tau from {grid} and the penalty together, per fold, on the validation tubes):")
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            say(Tt.round(3).to_string(index=False))

    # 2c' federated baselines, privacy, physics ablation, physical consistency
    def fmt(res):
        return "" if res is None else f"{res[0]:+.3f} [{res[1]:+.3f}, {res[2]:+.3f}]"

    def table(title, specs, fname, extra=None):
        rows_ = []
        for v, ref in specs:
            P = p(B, v)
            if P is None:
                continue
            r = {"method": label(v)}
            for H in (7, 30):
                r[f"auc_{H}d"] = auc_brier(P, H)[0]
                r[f"vs_{H}d"] = fmt(paired(P, p(B, ref), H, None, reps)) if ref and ref != v else ""
            r["vs_7d_logs_from_installation"] = \
                fmt(paired(P, p(B, ref), 7, full, reps)) if ref and ref != v and full else ""
            r["compared_with"] = label(ref) if ref and ref != v else ""
            if extra:
                r.update(extra(v, P))
            rows_.append(r)
        if len(rows_) > 1:
            Tb = pd.DataFrame(rows_)
            Tb.round(3).to_csv(os.path.join(out, fname), index=False)
            say(f"\n{title}")
            with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
                say(Tb.round(3).to_string(index=False))

    table("Federated baselines in B2 (same network, inputs, prior and loss; penalty 0.01 for every method; AUC "
          "difference vs FedAvg, 95% CI):",
          [(v, "fedavg") for v in ("fedavg", "dtg2", "dtg", "fedprox", "scaffold", "fedbn", "fedbn-dtg2", "cfl",
                                   "central", "nophys")],
          "table_federated_baselines.csv")
    table("Local normalization combined with trend weighting in B2 (FedBN with the DTG-v2 weights vs FedBN; penalty 0.01):",
          [("fedbn", None), ("fedbn-dtg2", "fedbn")], "table_fedbn_dtg2.csv")
    # controls: DTG-v2 against each baseline directly, and where its gain over FedAvg comes from
    table("DTG-v2 against each federated baseline directly in B2 (same fixed penalty 0.01; AUC difference DTG-v2 minus "
          "the baseline, 95% CI):",
          [("dtg2", ref) for ref in ("fedavg", "fedavg-eq", "dtg", "fedprox", "scaffold", "fedbn", "fedbn-dtg2", "cfl",
                                     "dtg2-shuf")],
          "table_dtg2_vs_baselines.csv")
    table("Source of the DTG-v2 gain in B2, fixed penalty 0.01 (equal client weights: does flattening the FedAvg weights "
          "explain it? shuffled similarities: does the trend information matter?):",
          [("fedavg", None), ("fedavg-eq", "fedavg"), ("dtg2-shuf", "fedavg"), ("dtg2", "fedavg"), ("dtg2", "fedavg-eq"),
           ("dtg2", "dtg2-shuf"), ("dtg", "fedavg-eq")],
          "table_dtg2_controls_fixed.csv")
    table("Final models of equal size in B2 (3 grid members each: the penalties 0.01, 0.1, 1; DTG-v2 at tau 0.05; "
          "ensemble + smoothing chosen on the validation tubes):",
          [("fedavg-ens-s", None), ("fedavg-eq-ens-s", "fedavg-ens-s"), ("dtg2-t0.05-ens-s", "fedavg-ens-s"),
           ("dtg2-t0.05-ens-s", "fedavg-eq-ens-s"), ("dtg2-shuf-ens-s", "fedavg-ens-s"),
           ("dtg2-t0.05-ens-s", "dtg2-shuf-ens-s"), ("dtg2-ens-s", "dtg2-t0.05-ens-s"), ("dtg2-t0.05-ens-s", "prior-s"),
           ("fedavg-eq-ens-s", "prior-s")],
          "table_dtg2_controls_final.csv")
    # diagnostics from the training log (no new training): CFL cluster split; DTG-v2 weights of late-starting logs
    try:
        TLg = json.load(open(os.path.join(B, "train_log.json"), encoding="utf-8"))
    except Exception:
        TLg = None
    try:
        warm = int(json.load(open(os.path.join(B, "config.json"), encoding="utf-8"))["args"].get("cfl_warmup", 5) or 5)
    except Exception:
        warm = 5
    if TLg:
        cur = {}
        for c in TLg.get("curves", []):
            cur[(c.get("variant"), str(c.get("fold")), int(c.get("seed", 0)))] = c
        cf = [c for (v, _, _), c in cur.items() if v == "cfl"]
        if cf:
            split = sum(1 for c in cf if c.get("clusters"))
            post = sum(1 for c in cf if c.get("clusters") and (c.get("best_round") or 0) > warm)
            say(f"\nCFL in B2: the clients were split into two clusters in {split} of {len(cf)} fold-seed runs; the model "
                f"kept (lowest validation loss) was trained after the split (best round > {warm}) in {post} of {len(cf)} runs; "
                "in the other runs CFL's prediction is that of the single FedAvg-trained model before the split.")
        P0b = p(B, "prior")
        if P0b is not None and full:
            nk = P0b.groupby("tube").size().to_dict()
            rows_w = []
            for e in TLg.get("dtg", []):
                v = e.get("variant")
                if not (isinstance(v, str) and v.startswith("dtg2")) or "-dp" in v or re.search(r"-r[0-9.]+$", v):
                    continue
                w = {k[2:]: float(x) for k, x in e.items() if k.startswith("w_") and k[2:] in nk}
                if len(w) < 3:
                    continue
                tot = sum(nk[t] for t in w)
                for t, x in w.items():
                    rows_w.append({"variant": v, "tube": t, "from_installation": t in full,
                                   "ratio_to_fedavg": x / (nk[t] / tot)})
            if rows_w:
                Wd = pd.DataFrame(rows_w)
                g = Wd.groupby(["variant", "from_installation"]).ratio_to_fedavg.agg(["mean", "median", "size"]).reset_index()
                g.round(3).to_csv(os.path.join(out, "table_dtg2_weight_by_log_start.csv"), index=False)
                say("\nDTG-v2 weight of a client relative to its FedAvg weight, by whether its log starts at installation "
                    "(mean and median over folds and seeds; 1 = same as FedAvg):")
                say(g.round(3).to_string(index=False))
        # Where the gain over FedAvg comes from. With one local epoch per round a client's number of local
        # steps grows with its samples n_k, so its effective weight in the update is (aggregation weight) x (local
        # steps) (Wang et al., 2020): FedAvg ~ n_k^2, equal weights ~ n_k. Compared over the same training clients.
        try:
            Pall = read(os.path.join(B, "pred_mean__prior.csv"))
            try:
                batch = int(json.load(open(os.path.join(B, "config.json"), encoding="utf-8"))["args"].get("batch", 32) or 32)
            except Exception:
                batch = 32
            if Pall is not None:
                nk_all = Pall.groupby("tube").size().astype(float).to_dict()
                fams = {"dtg2": "DTG-v2 (tau 0.05)", "dtg2-shuf": "DTG-v2, shuffled similarities (tau 0.05)",
                        "dtg": "DTG, original weights"}
                last = {}
                for e in TLg.get("dtg", []):                 # the last entry per run is the current one
                    if e.get("variant") in fams:
                        last[(e["variant"], str(e.get("fold")), int(e.get("seed", 0)))] = e

                def stats(x, ns, scheme, fold, seed, factor=None):
                    ms = np.ceil(ns / batch)
                    eff = x * ms / np.sum(x * ms)
                    r = {"scheme": scheme, "fold": fold, "seed": seed, "largest_share": float(eff.max()),
                         "longest_log_share": float(eff[int(np.argmax(ns))]), "n_eff": float(1.0 / np.sum(eff ** 2))}
                    if factor is not None:               # rank correlation of the trend factor with the log length
                        rf, rn = pd.Series(factor).rank().to_numpy(), pd.Series(ns).rank().to_numpy()
                        r["factor_vs_length_spearman"] = float(np.corrcoef(rf, rn)[0, 1]) if np.std(rf) > 0 else np.nan
                        r["longest_log_factor"] = float(factor[int(np.argmax(ns))])
                    return r

                rows_e = []
                for (v, fold, seed), e in last.items():
                    w = {k[2:]: float(x) for k, x in e.items() if k.startswith("w_") and k[2:] in nk_all}
                    if len(w) < 3:
                        continue
                    ts = sorted(w)
                    ns = np.array([nk_all[t] for t in ts])
                    x = np.array([w[t] for t in ts])
                    x = x / x.sum()
                    fed = ns / ns.sum()
                    rows_e.append(stats(x, ns, fams[v], fold, seed, factor=x / fed))
                    if v == "dtg2":                          # FedAvg and equal weights over the same training clients
                        rows_e.append(stats(fed, ns, "FedAvg (weights ~ n_k)", fold, seed))
                        rows_e.append(stats(np.full(len(ns), 1.0 / len(ns)), ns, "FedAvg, equal client weights",
                                            fold, seed))
                if rows_e:
                    Ed = pd.DataFrame(rows_e)
                    Ed.to_csv(os.path.join(out, "effective_weights_by_run.csv"), index=False)
                    order = ["FedAvg (weights ~ n_k)", "FedAvg, equal client weights", fams["dtg"], fams["dtg2"],
                             fams["dtg2-shuf"]]
                    agg = {"runs": ("fold", "size"), "largest_share": ("largest_share", "mean"),
                           "longest_log_share": ("longest_log_share", "mean"), "n_eff": ("n_eff", "mean")}
                    if "factor_vs_length_spearman" in Ed:
                        agg.update({"factor_vs_length_spearman": ("factor_vs_length_spearman", "mean"),
                                    "longest_log_factor": ("longest_log_factor", "mean")})
                    g = Ed.groupby("scheme").agg(**agg).reindex([s for s in order if s in set(Ed.scheme)]).reset_index()
                    g.round(3).to_csv(os.path.join(out, "table_effective_weights.csv"), index=False)
                    say("\nEffective client weights in B2 (aggregation weight x local steps; with one local epoch the steps "
                        "grow with the client's samples, Wang et al. 2020): share of the largest client and of the longest "
                        "log in each update, effective number of clients, and for the trend weightings the rank "
                        "correlation of the trend factor (weight / FedAvg weight) with the log length and the factor of the "
                        "longest log (mean over folds and seeds; same training clients for every scheme):")
                    with pd.option_context("display.width", 250, "display.max_columns", 20):
                        say(g.round(3).to_string(index=False))
        except Exception as ex:                          # a diagnostic must never stop the report
            say(f"\n(effective client weights skipped: {type(ex).__name__}: {ex})")
    S_ = read(os.path.join(B, "summary.csv"))
    eps_of = {} if S_ is None or "dp_epsilon" not in S_.columns else \
        S_.dropna(subset=["dp_epsilon"]).groupby("variant").dp_epsilon.first().to_dict()
    table("Privacy in B2 (the DP-FedAvg setting without noise vs FedAvg; client-level DP-FedAvg vs that setting without "
          "noise; DTG-v2 with its trend released under Laplace DP vs DTG-v2):",
          [("fedavg", None), ("cdp0", "fedavg"), ("cdp0.5", "cdp0"), ("cdp1", "cdp0"), ("cdp2", "cdp0"), ("dtg2", None),
           ("dtg2-dp1", "dtg2"), ("dtg2-dp0.1", "dtg2")], "table_privacy.csv",
          extra=lambda v, P: {"epsilon": eps_of.get(v, float(v.split("-dp")[1]) if "-dp" in v else np.nan)})
    table("Physics ablation in B2 (difference vs the full FedAvg model, penalty 0.01) and physical consistency "
          "(share of day pairs >= 30 days apart, similar workload, in which the workload-normalised cumulative hazard "
          "-ln(1-P)/(scans per day) is more than 5% lower on the later day; wear-out says it should not fall):",
          [(v, "fedavg") for v in ("prior", "nophys", "fedavg-m0", "fedavg", "fedavg-tuned", "dtg2-tuned", "fedavg-ens-s",
                                   "dtg2-ens-s", "central-tuned")], "table_physics.csv",
          extra=lambda v, P: {"hazard_decrease_share_30d": mono_violation(P, 30),
                              "hazard_decrease_share_7d": mono_violation(P, 7)})
    # interval for the physical-consistency difference (data-only network minus physics-guided model)
    try:
        rows_m = []
        for xa, xb in (("nophys", "fedavg"), ("nophys", "central-tuned"), ("nophys-s", "fedavg-ens-s"),
                       ("nophys-s", "dtg2-ens-s"), ("nophys-s", "central-ens-s")):
            Pa, Pb = p(B, xa), p(B, xb)
            if Pa is None or Pb is None:
                continue
            for H in (30, 7):
                r = mono_delta(Pa, Pb, H, reps)
                if r:
                    rows_m.append({"model_a": label(xa), "model_b": label(xb), "horizon_days": H, "share_a": r[0],
                                   "share_b": r[1], "difference": r[2], "ci_low": r[3], "ci_high": r[4], "tubes": r[5]})
        if rows_m:
            Tm = pd.DataFrame(rows_m)
            Tm.round(4).to_csv(os.path.join(out, "table_physics_consistency_ci.csv"), index=False)
            say("\nPhysical consistency, data-only network minus physics-guided model (share of day pairs with a falling "
                "hazard as above, the same pairs for both models; difference with 95% CI from a tube-level bootstrap):")
            with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
                say(Tm.round(3).to_string(index=False))
    except Exception as ex:                          # a diagnostic must never stop the report
        say(f"\n(physical-consistency intervals skipped: {type(ex).__name__}: {ex})")
    st = read(os.path.join(B, "dtg_stability.csv"))
    if st is not None and not st.empty:
        st.round(4).to_csv(os.path.join(out, "table_dtg_stability.csv"), index=False)
        say("\nDTG weights from incomplete trajectories (truncation: share of each log available) and with the trend "
            "under Laplace DP (setting: epsilon); L1 distance and rank correlation to the weights from the full logs:")
        say(st.round(4).to_string(index=False))
    for fn_, title in (("table_complexity_model.csv", "Complexity of the network (per prediction = one tube-day):"),
                       ("table_complexity_training.csv", "Training time and communication (rounds actually run):")):
        cx = read(os.path.join(out, fn_))
        if cx is not None and not cx.empty:
            say(f"\n{title}")
            with pd.option_context("display.width", 250, "display.max_columns", 20):
                say(cx.to_string(index=False))

    # 2c training length actually used in B2 (early stopping on the validation tubes)
    tl = read(os.path.join(B, "training_length.csv"))
    if tl is not None and not tl.empty:
        g = tl.groupby("variant").agg(runs=("rounds_run", "size"), rounds_run_median=("rounds_run", "median"),
                                      best_round_median=("best_round", "median"),
                                      still_improving_at_end=("still_improving_at_end", "mean")).reset_index()
        g.round(3).to_csv(os.path.join(out, "table_training_length.csv"), index=False)
        say("\nTraining length in B2 (rounds; centralized: epochs). 'still improving at end' = share of runs stopped by the "
            "maximum while the validation loss was still falling (should be near 0):")
        say(g.round(2).to_string(index=False))

    # 3a which penalty the nested selection chose
    tc = read(os.path.join(B, "tuning_choices.csv"))
    if tc is not None and not tc.empty:
        g = tc.groupby(["tuned", "chosen"]).size().rename("fold_seed_runs").reset_index()
        g.insert(2, "setting", g.chosen.map(label))
        g.to_csv(os.path.join(out, "table_tuning_choices.csv"), index=False)
        say("\nNested selection in B2 (setting chosen on the validation tubes of each fold; number of fold-seed runs):")
        say(g.to_string(index=False))
    # 3 DTG-v2 weights actually used (effective number of clients vs FedAvg)
    lg = os.path.join(B, "train_log.json")
    if os.path.exists(lg):
        L = json.load(open(lg, encoding="utf-8"))
        W = pd.DataFrame([x for x in L.get("dtg", []) if "n_eff" in x])
        if not W.empty:
            g = W.groupby("variant").agg(folds=("fold", "count"), w_min=("w_min", "min"), w_max=("w_max", "max"),
                                         n_eff=("n_eff", "mean"), n_eff_fedavg=("n_eff_fedavg", "mean")).reset_index()
            g.round(4).to_csv(os.path.join(out, "table_dtg2_weights.csv"), index=False)
            say("\nDTG-v2 aggregation weights in B2 (effective number of clients = 1 / sum of squared weights):")
            say(g.round(3).to_string(index=False))

    # 4 figure
    if not O.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        inputs = list(dict.fromkeys(O["input"]))
        cmap = plt.get_cmap("tab10")
        for ax, col, title in zip(axes, ("auc_30d", "auc_7d"), ("Failure within 30 days", "Failure within 7 days")):
            show = [v for v in ("prior", "prior-s", "nophys", "fedavg-tuned", "dtg2-tuned", "fedavg-ens-s", "dtg2-ens-s",
                                "central-ens-s", "fedavg", "dtg", "central") if v in set(O.variant)]
            meths = [label(v) for v in show]
            x = np.arange(len(meths))
            for i, inp in enumerate(inputs):
                e = O[O["input"] == inp].set_index("method").reindex(meths)
                ax.bar(x + (i - len(inputs) / 2 + 0.5) * 0.8 / len(inputs), e[col], 0.8 / len(inputs),
                       color=cmap(i), label=inp)
            ax.set_xticks(x)
            ax.set_xticklabels(meths, rotation=30, ha="right", fontsize=7.5)
            ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
            ax.set_ylim(0.4, 1.0)
            ax.set(ylabel="AUC (leave one tube out)", title=title)
        axes[0].legend(fontsize=6.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"fig_auc_by_input.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        say("\nfigure fig_auc_by_input written")
    open(os.path.join(out, "comparison_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"written to {out}: comparison_report.txt, table_*.csv, criteria.csv, fig_auc_by_input")


if __name__ == "__main__":
    main()
