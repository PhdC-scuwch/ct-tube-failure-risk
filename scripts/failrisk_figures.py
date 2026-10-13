#!/usr/bin/env python3
"""Figures and tables for the failure-probability study (reads the outputs of scripts/failrisk.py).

  python scripts/failrisk_figures.py --loo <results>/loo --loho <results>/loho --site-map site_map.csv --out <results>/figures

Tube names are anonymised by hospital (A01, B01, ...; largest hospital = A). tube_labels.csv maps
the labels to the real tubes (keep it private).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from dtgexp.failrisk import metrics, tube_weights  # noqa: E402
from dtgexp.survival import fit_weibull  # noqa: E402

LAB = {"prior": "Physics prior only (Weibull + workload)", "fedavg": "Proposed, FedAvg",
       "dtg": "Proposed, DTG (original weighting)", "central": "Proposed, centralized", "nophys": "Data-only network (no physics)",
       "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedbn": "FedBN", "cfl": "CFL"}
COL = {"prior": "#7F8C8D", "fedavg": "#2E86C1", "dtg": "#C0392B", "central": "#1E8449", "nophys": "#B9770E",
       "fedprox": "#8E44AD", "scaffold": "#16A085", "fedbn": "#D35400", "cfl": "#5D6D7E"}


def lab(v):
    if v.endswith("-s"):
        return lab(v[:-len("-s")]) + " + smoothing"
    if v.endswith("-ens"):
        b = v[:-len("-ens")]
        return f"Proposed, {({'fedavg': 'FedAvg', 'dtg2': 'DTG-v2', 'central': 'centralized'}).get(b, b)} (ensemble)"
    if re.fullmatch(r".+-m[0-9.]+", v):
        b, m = v.rsplit("-m", 1)
        return f"{lab(b)}, " + ("no monotonicity loss" if float(m) == 0 else f"monotonicity weight {m}")
    if v.startswith("dtg2-dp"):
        return f"DTG-v2, trend DP eps={v[7:]}"
    if v.startswith("dtg-dp"):
        return f"DTG, trend DP eps={v[6:]}"
    if v in LAB:
        return LAB[v]
    if v == "fedavg-tuned":
        return "Proposed, FedAvg (tuned on validation tubes)"
    if v == "dtg2-tuned":
        return "Proposed, DTG-v2 (tuned on validation tubes)"
    if v.startswith("dtg2-t") and v.endswith("-tuned"):
        return f"DTG-v2, tau {v[len('dtg2-t'):-len('-tuned')]} (penalty tuned)"
    if v == "central-tuned":
        return "Proposed, centralized (tuned on validation tubes)"
    if "-r" in v and v.rsplit("-r", 1)[1].replace(".", "", 1).isdigit():
        b, r = v.rsplit("-r", 1)
        return f"{lab(b)}, penalty {r}"
    if v.startswith("dtg2"):
        return "DTG-v2, tau " + (v.split("-t")[1] if "-t" in v else "0.05")
    if v.startswith("cdp"):
        return f"Client-level DP, z={v[3:]}"
    return v


def col(v):
    if v.endswith("-s"):
        return col(v[:-len("-s")])
    if v.endswith("-ens"):
        return {"fedavg": "#154360", "dtg2": "#4A0E0E", "central": "#0B3D20"}.get(v[:-len("-ens")], "#333333")
    if v == "fedavg-tuned":
        return "#1B4F72"
    if v == "dtg2-tuned":
        return "#641E16"
    if v == "central-tuned":
        return "#145A32"
    if "-r" in v and v.rsplit("-r", 1)[1].replace(".", "", 1).isdigit():
        return {"fedavg": "#AED6F1", "central": "#A9DFBF"}.get(v.rsplit("-r", 1)[0], "#F5B7B1")
    if v.startswith("dtg2"):
        return {"dtg2": "#7B241C", "dtg2-t0.02": "#A93226", "dtg2-t0.1": "#E6B0AA"}.get(v, "#922B21")
    return COL.get(v, "#AAB7B8" if v.startswith("cdp") else "#E59866")


def save(fig, out, name, made):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    made.append(name)
    print(f"  figure {name}")


def table(df, out, name, made):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    open(os.path.join(out, f"{name}.md"), "w", encoding="utf-8").write(md + "\n")
    made.append(name)
    print(f"  table  {name}")


def anonymize(tubes, site_map):
    hosp = {}
    if site_map and os.path.exists(site_map):
        t = pd.read_csv(site_map, dtype=str, encoding="utf-8-sig").fillna("")
        t.columns = [c.strip().lower() for c in t.columns]
        m = {str(r.tube).replace("device_", "").strip(): str(r.hospital).strip() for r in t.itertuples()}
        hosp = {n: m.get(n.replace("device_", ""), "?") for n in tubes}
    if not hosp:
        return {n: f"T{i + 1:02d}" for i, n in enumerate(sorted(tubes))}, {}
    counts = pd.Series(hosp).value_counts()
    order = sorted(counts.index, key=lambda h: (-counts[h], h))
    letter = {h: chr(65 + i) for i, h in enumerate(order)}
    names, k = {}, {}
    for n in sorted(tubes, key=lambda n: (letter[hosp[n]], n)):
        L = letter[hosp[n]]
        k[L] = k.get(L, 0) + 1
        names[n] = f"{L}{k[L]:02d}"
    return names, letter


def load_preds(folder):
    P = {}
    for f in glob.glob(os.path.join(folder, "pred_mean__*.csv")):
        P[os.path.basename(f)[len("pred_mean__"):-4]] = pd.read_csv(f)
    return P


def fig_roc(P, out, made):
    show = [v for v in ("prior-s", "nophys-s", "fedavg-ens-s", "dtg2-ens-s", "central-ens-s", "prior", "nophys",
                        "fedavg-tuned", "dtg2-tuned", "central-tuned", "fedavg", "dtg", "dtg2", "central") if v in P][:6]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, H in zip(axes, (30, 7)):
        for v in show:
            d = P[v]
            w = tube_weights(d.tube.to_numpy())
            fpr, tpr, _ = roc_curve(d[f"y{H}"], d[f"p{H}"], sample_weight=w)
            auc = metrics(d[f"y{H}"].to_numpy(), d[f"p{H}"].to_numpy(), w)["auc"]
            ax.plot(fpr, tpr, color=col(v), lw=1.6, label=f"{lab(v)} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], ":", color="#999999", lw=0.8)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Failure within {H} days")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig1_roc", made)


def fig_calibration(P, out, made):
    show = [v for v in ("prior-s", "fedavg-ens-s", "dtg2-ens-s", "prior", "fedavg-tuned", "dtg2-tuned", "fedavg", "dtg")
            if v in P][:3]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, H in zip(axes, (30, 7)):
        for v in show:
            d = P[v]
            w = tube_weights(d.tube.to_numpy())
            p, y = d[f"p{H}"].to_numpy(), d[f"y{H}"].to_numpy()
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            k = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
            xs, ys = [], []
            for b in range(len(edges) - 1):
                m = k == b
                if m.sum() > 20:
                    xs.append(np.average(p[m], weights=w[m]))
                    ys.append(np.average(y[m], weights=w[m]))
            ax.plot(xs, ys, "o-", color=col(v), lw=1.4, ms=3.5, label=lab(v))
        lim = max(0.05, max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.plot([0, lim], [0, lim], ":", color="#999999", lw=0.8)
        ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title=f"Calibration, {H} days")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig2_calibration", made)


def fig_timelines(P, names, out, made):
    main = next((v for v in ("dtg2-ens-s", "fedavg-ens-s", "dtg2-tuned", "fedavg-tuned", "dtg", "fedavg") if v in P), None)
    if main is None or "prior" not in P:
        return
    life = P["prior"].groupby("tube").dtf_days.max().sort_values()
    pick = list(dict.fromkeys([life.index[0], life.index[len(life) // 3], life.index[2 * len(life) // 3], life.index[-1]]))
    fig, axes = plt.subplots(len(pick), 1, figsize=(8, 2.0 * len(pick)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, pick):
        ref = "prior-s" if main.endswith("-s") and "prior-s" in P else "prior"
        for v, ls in ((ref, "--"), (main, "-")):
            d = P[v][P[v].tube == t].sort_values("dtf_days", ascending=False)
            ax.plot(-d.dtf_days, d.p30, ls, color=col(v), lw=1.3, label=f"{lab(v)}, 30 d")
            ax.plot(-d.dtf_days, d.p7, ls, color=col(v), lw=0.9, alpha=0.6, label=f"{lab(v)}, 7 d")
        ax.axvspan(-30, 0, color="#F5B7B1", alpha=0.35, lw=0)
        ax.set_ylabel(names.get(t, t))
        ax.set_xlim(-min(life.max(), 400), 2)
    axes[0].legend(fontsize=6.5, ncol=2, loc="upper left")
    axes[-1].set_xlabel("Days relative to failure (shaded: last 30 days)")
    fig.tight_layout()
    save(fig, out, "fig3_example_tubes", made)


def fig_baselines(T, out, made):
    if T is None or T.empty:
        return
    d = T[T["compare"].str.endswith(" - prior")].copy()
    d["variant"] = d["compare"].str.replace(" - prior", "", regex=False)
    tuned_here = {v[:-len("-tuned")] for v in set(d.variant) if v.endswith("-tuned")}

    def member(v):                                   # fixed-setting grid member of a tuned variant shown here
        if v.endswith("-tuned"):
            return False
        b = re.sub(r"-r[0-9.]+$", "", v)
        return ("dtg2" in tuned_here or b in tuned_here) if b.startswith("dtg2") else b in tuned_here
    order = [v for v in ("nophys", "fedavg-tuned", "dtg2-tuned", "central-tuned", "fedavg", "dtg", "dtg2", "central", "fedprox",
                         "scaffold", "fedbn", "cfl") if v in set(d.variant) and not member(v)]
    order += sorted((v for v in set(d.variant) if v not in order and not member(v)),
                    key=lambda v: (not v.startswith("dtg2-t"), v))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, H in zip(axes, (30, 7)):
        e = d[d.horizon_days == H].set_index("variant").reindex(order)
        y = np.arange(len(order))
        ax.errorbar(e.delta_auc, y, xerr=[e.delta_auc - e.ci_low, e.ci_high - e.delta_auc], fmt="o", color="#333333",
                    ecolor="#888888", capsize=2, ms=4)
        for i, v in enumerate(order):
            ax.plot(e.delta_auc.iloc[i], i, "o", color=col(v), ms=6)
        ax.axvline(0, color="#999999", lw=0.8, ls=":")
        ax.axvline(0.05, color="#C0392B", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([lab(v) for v in order], fontsize=7.5)
        ax.set(xlabel="AUC difference vs physics prior (95% CI)", title=f"Failure within {H} days")
    fig.tight_layout()
    save(fig, out, "fig4_federated_baselines", made)


def fig_convergence(loo, out, made):
    """Validation loss per round and the round of the model kept: drawn by scripts/plot_convergence.py."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_convergence import make
    if make(loo, out):
        made.append("fig5_convergence")
        print("  figure fig5_convergence")


def fig_dtg(loo, names, out, made):
    sim_f, w_f, st_f = (os.path.join(loo, f) for f in ("dtg_similarity_full.csv", "dtg_weights_full.csv", "dtg_stability.csv"))
    if not os.path.exists(sim_f):
        return
    R = pd.read_csv(sim_f, index_col=0)
    lbl = [names.get(n, n) for n in R.index]
    order = np.argsort(lbl)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    im = axes[0].imshow(R.to_numpy()[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(lbl)))
    axes[0].set_yticks(range(len(lbl)))
    axes[0].set_xticklabels(np.array(lbl)[order], rotation=90, fontsize=6)
    axes[0].set_yticklabels(np.array(lbl)[order], fontsize=6)
    axes[0].set_title("Trend similarity (stage-aligned, shared stage range)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    W = pd.read_csv(w_f)
    W["label"] = [names.get(n, n) for n in W.tube]
    W = W.sort_values("label")
    x = np.arange(len(W))
    axes[1].bar(x - 0.2, W.weight_stage, 0.4, color=col("dtg"), label="stage-aligned")
    axes[1].bar(x + 0.2, W.weight_index, 0.4, color="#AAB7B8", label="index-based (original)")
    axes[1].axhline(1 / len(W), color="#333333", lw=0.8, ls=":", label="uniform")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(W.label, rotation=90, fontsize=6)
    axes[1].set(ylabel="Aggregation weight", title="DTG weights (all tubes)")
    axes[1].legend(fontsize=7)
    if os.path.exists(st_f):
        S = pd.read_csv(st_f)
        t = S[S.analysis == "truncation"]
        kinds = [("stage-aligned", col("dtg")), ("index-based (original)", "#7F8C8D")] + \
            [(k, col("dtg2")) for k in sorted(set(t.trend)) if k.startswith("DTG-v2")]
        for kind, c in kinds:
            e = t[t.trend == kind].sort_values("setting")
            if not e.empty:
                axes[2].plot(e.setting * 100, e.L1_to_full, "o-", color=c, label=kind)
        axes[2].set(xlabel="Share of each trajectory available (%)", ylabel="L1 distance to full-trajectory weights",
                    title="Weights from incomplete trajectories")
        axes[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig6_dtg_trend", made)


def fig_dp(loo, S, out, made):
    st_f = os.path.join(loo, "dtg_stability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if os.path.exists(st_f):
        d = pd.read_csv(st_f)
        d = d[d.analysis == "dp"]
        v2 = d[d.trend.str.startswith("DTG-v2")]
        d = (v2 if not v2.empty else d[d.trend == d.trend.iloc[0]] if not d.empty else d).sort_values("setting")
        axes[0].semilogx(d.setting, d.L1_to_full, "o-", color=col("dtg"), label="L1 distance")
        ax2 = axes[0].twinx()
        ax2.semilogx(d.setting, d.spearman_to_full, "s--", color="#7F8C8D", label="rank correlation")
        axes[0].set(xlabel="Trend privacy budget epsilon", ylabel="L1 distance to non-private weights",
                    title="DTG weights under Laplace DP")
        ax2.set_ylabel("Spearman correlation with non-private weights")
    if S is not None:
        e = S[S.horizon_days == 30].set_index("variant")
        vs = [v for v in e.index if v in ("fedavg", "dtg", "dtg2") or v.startswith(("dtg-dp", "dtg2-dp", "cdp"))]
        vals = [e.loc[v, "auc"] for v in vs]
        axes[1].bar(range(len(vs)), vals, color=[col(v) for v in vs])
        for i, v in enumerate(vs):
            ep = e.loc[v, "dp_epsilon"] if "dp_epsilon" in e.columns else np.nan
            txt = f"eps={ep:.0f}" if np.isfinite(ep) else ""
            axes[1].text(i, vals[i] + 0.005, txt, ha="center", fontsize=7)
        axes[1].set_xticks(range(len(vs)))
        axes[1].set_xticklabels([lab(v) for v in vs], rotation=30, ha="right", fontsize=7)
        lo = min(vals) - 0.05 if vals else 0
        axes[1].set_ylim(max(0.0, lo), 1.0)
        axes[1].set(ylabel="AUC, failure within 30 days", title="Utility under differential privacy")
    fig.tight_layout()
    save(fig, out, "fig7_privacy", made)


def fig_loho(loho, out, made):
    P = load_preds(loho)
    if not P:
        return None
    rows = []
    for v, d in P.items():
        for f, g in d.groupby("fold"):
            for H in (7, 30):
                m = metrics(g[f"y{H}"].to_numpy(), g[f"p{H}"].to_numpy(), tube_weights(g.tube.to_numpy()))
                rows.append({"held_out_hospital": f, "tubes": g.tube.nunique(), "variant": v, "horizon_days": H,
                             "auc": m["auc"], "brier": m["brier"]})
    D = pd.DataFrame(rows)
    hs = sorted(D.held_out_hospital.unique())
    vs = [v for v in ("prior-s", "nophys-s", "fedavg-ens-s", "dtg2-ens-s", "central-ens-s", "prior", "nophys",
                      "fedavg-tuned", "dtg2-tuned", "central-tuned", "dtg", "central") if v in set(D.variant)][:6]
    if not any(v.endswith("-tuned") for v in vs):
        vs = [v for v in ("prior", "nophys", "fedavg", "dtg", "dtg2", "central") if v in set(D.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, H in zip(axes, (30, 7)):
        for i, v in enumerate(vs):
            e = D[(D.variant == v) & (D.horizon_days == H)].set_index("held_out_hospital").reindex(hs)
            ax.bar(np.arange(len(hs)) + (i - len(vs) / 2 + 0.5) * 0.8 / len(vs), e.auc, 0.8 / len(vs),
                   color=col(v), label=lab(v))
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels([h.replace("_", " ") for h in hs])
        ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
        ax.set(ylabel="AUC", title=f"Leave one hospital out, {H} days")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig8_leave_one_hospital_out", made)
    return D


def fig_weibull(loo, names, out, made):
    files = glob.glob(os.path.join(loo, "cache", "*.npz")) or glob.glob(os.path.join(os.path.dirname(loo), "**", "*.npz"), recursive=True)
    lives, days = {}, {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        n = os.path.basename(f).split("__")[0]
        if float(z["life"]) >= 5000:
            lives[n], days[n] = float(z["life"]), float(z["days"]) * float(z["life"]) / max(float(z["sc"][-1] - z["sc"][0]), 1.0)
    if len(lives) < 4:
        return None
    L = np.array(sorted(lives.values()))
    b, e = fit_weibull(L)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].step(L, np.arange(1, len(L) + 1) / len(L), where="post", color="#333333", label="tubes")
    x = np.linspace(0, L.max() * 1.1, 300)
    axes[0].plot(x, 1 - np.exp(-(x / e) ** b), color=col("prior"), lw=1.6, label=f"Weibull beta={b:.2f}, eta={e:,.0f}")
    axes[0].set(xlabel="Life (scans)", ylabel="Cumulative share failed", title="Life distribution in scans")
    axes[0].legend(fontsize=7)
    cv_s = np.std(L, ddof=1) / np.mean(L)
    Dd = np.array(list(days.values()))
    cv_d = np.std(Dd, ddof=1) / np.mean(Dd)
    axes[1].bar([0, 1], [cv_s, cv_d], color=[col("prior"), "#AAB7B8"])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["life in scans", "life in days\n(estimated from logged span)"])
    axes[1].set(ylabel="Coefficient of variation", title="Which clock ages the tube?")
    fig.tight_layout()
    save(fig, out, "fig9_weibull_prior", made)
    return pd.DataFrame({"tube": [names.get(n, n) for n in lives], "life_scans": list(lives.values()),
                         "life_days_est": [days[n] for n in lives]}).sort_values("tube")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loo", required=True)
    ap.add_argument("--loho", default="")
    ap.add_argument("--site-map", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    made = []
    P = load_preds(a.loo)
    if not P:
        raise SystemExit(f"no pred_mean__*.csv in {a.loo}")
    tubes = sorted(set(next(iter(P.values())).tube))
    names, _ = anonymize(tubes, a.site_map)
    pd.DataFrame({"label": [names[t] for t in tubes], "tube": tubes}).to_csv(os.path.join(a.out, "tube_labels.csv"), index=False)
    S = pd.read_csv(os.path.join(a.loo, "summary.csv")) if os.path.exists(os.path.join(a.loo, "summary.csv")) else None
    T = pd.read_csv(os.path.join(a.loo, "tests.csv")) if os.path.exists(os.path.join(a.loo, "tests.csv")) else None

    def step(name, fn, *args):                       # announce each figure first: a stall shows where it is
        print(f"  making {name} ...", flush=True)
        return fn(*args)
    step("fig1_roc", fig_roc, P, a.out, made)
    step("fig2_calibration", fig_calibration, P, a.out, made)
    step("fig3_example_tubes", fig_timelines, P, names, a.out, made)
    step("fig4_federated_baselines", fig_baselines, T, a.out, made)
    if os.path.exists(os.path.join(a.loo, "train_log.json")):
        step("fig5_convergence", fig_convergence, a.loo, a.out, made)
    step("fig6_dtg_trend", fig_dtg, a.loo, names, a.out, made)
    step("fig7_privacy", fig_dp, a.loo, S, a.out, made)
    D = step("fig8_leave_one_hospital_out", fig_loho, a.loho, a.out, made) if a.loho and os.path.isdir(a.loho) else None
    life = step("fig9_weibull_prior", fig_weibull, a.loo, names, a.out, made)
    if S is not None:
        M = S.copy()
        M.insert(0, "method", [lab(v) for v in M.variant])
        if T is not None:
            d = T[T["compare"].str.endswith(" - prior")].copy()
            d["variant"] = d["compare"].str.replace(" - prior", "", regex=False)
            M = M.merge(d[["variant", "horizon_days", "delta_auc", "ci_low", "ci_high"]], on=["variant", "horizon_days"], how="left")
        table(M.round(4), a.out, "table_main", made)
    if T is not None:
        table(T.round(4), a.out, "table_auc_differences", made)
    if os.path.exists(os.path.join(a.loo, "communication.csv")):
        C = pd.read_csv(os.path.join(a.loo, "communication.csv"))
        C.insert(0, "method", [lab(v) for v in C.variant])
        table(C.round(3), a.out, "table_communication", made)
    if os.path.exists(os.path.join(a.loo, "dtg_stability.csv")):
        table(pd.read_csv(os.path.join(a.loo, "dtg_stability.csv")).round(4), a.out, "table_dtg_stability", made)
    if D is not None:
        table(D.round(4), a.out, "table_leave_one_hospital_out", made)
    if life is not None:
        table(life.round(1), a.out, "table_tube_lives", made)
    print(f"done: {len(made)} items in {a.out}; tube_labels.csv maps labels to real tubes (keep it private)")


if __name__ == "__main__":
    main()
