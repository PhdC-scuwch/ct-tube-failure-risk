#!/usr/bin/env python3
"""Convergence figure (no training; reads <loo>/train_log.json and <loo>/config.json):

  (a) median validation loss per communication round: DTG-v2 (with the interquartile range over its runs) and
      FedAvg in colour, the other federated methods (penalty 0.01) in grey;
  (b) for every method, the round at which each run reached its lowest validation loss (the model that was kept).

  python scripts/plot_convergence.py --loo results/B2/loo

Writes fig5_convergence.png / .pdf and convergence_summary.csv to <loo>/../figures (or --out) and prints the
summary. Only numpy and matplotlib are used (no torch), so it runs on its own in a few seconds;
scripts/failrisk_figures.py draws its convergence figure with the same function.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

# variant key in the runner and its label; the methods in the order of panel (b)
METHODS = [
    ("dtg2", "DTG-v2 (proposed)"),
    ("fedavg", "FedAvg"),
    ("dtg", "Original DTG weighting"),
    ("fedprox", "FedProx"),
    ("scaffold", "SCAFFOLD"),
    ("fedbn", "FedBN"),
    ("fedbn-dtg2", "FedBN + DTG-v2 weights"),
    ("cfl", "Clustered FL"),
    ("central", "Centralized (per epoch)"),
    ("nophys", "Data-only network"),
]
# emphasis: the proposed method and its direct comparator in colour (two categorical slots that pass the
# colour-vision checks on white), every other method in grey; text stays in ink, never in a series colour
HIGHLIGHT = {"dtg2": "#2a78d6", "fedavg": "#eb6834"}
GREY_LINE, GREY_DOT = "#bdbcb5", "#9d9c95"
INK, INK2, AXIS, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load_runs(loo):
    """The current run of every (variant, fold, seed): reruns append to the log, so the last entry is kept."""
    with open(os.path.join(loo, "train_log.json"), encoding="utf-8") as fh:
        log = json.load(fh)
    runs = {}
    for c in log.get("curves", []):
        cv = c.get("curve") or []
        if any(x is not None for x in cv):
            runs[(c["variant"], str(c["fold"]), int(c["seed"]))] = c
    cfg = {}
    try:
        with open(os.path.join(loo, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh).get("args", {})
    except (OSError, ValueError):
        pass
    return runs, cfg


def curves_of(runs, variant):
    out = []
    for (v, _, _), c in sorted(runs.items()):
        if v == variant:
            a = np.array([np.nan if x is None else float(x) for x in c["curve"]], float)
            a[~np.isfinite(a)] = np.nan
            out.append(a)
    return out


def per_round_stats(C, min_share=0.5):
    """Median and quartiles over the runs still training at each evaluation; only evaluations that at least
    min_share of the runs reached (early stopping gives runs different lengths)."""
    L = max(len(c) for c in C)
    S = np.full((len(C), L), np.nan)
    for i, c in enumerate(C):
        S[i, :len(c)] = c
    alive = np.isfinite(S).sum(0)
    keep = alive >= min_share * len(C)
    with np.errstate(all="ignore"):
        q1, med, q3 = (np.nanpercentile(S, q, axis=0) for q in (25, 50, 75))
    return np.arange(1, L + 1)[keep], med[keep], q1[keep], q3[keep]


def round_ticks(axis, upper):
    """Integer round ticks at a step that suits the range (1, 2 or 5 rounds)."""
    axis.xaxis.set_major_locator(MultipleLocator(1 if upper <= 6 else 2 if upper <= 14 else 5))


def make(loo, out, methods=None, name="fig5_convergence"):
    """Draw the figure into <out>/<name>.png/.pdf, write convergence_summary.csv there and print the summary."""
    os.makedirs(out, exist_ok=True)
    print(f"reading {os.path.join(loo, 'train_log.json')}", flush=True)
    runs, cfg = load_runs(loo)
    local_epochs = int(cfg.get("local_epochs", 1) or 1)
    min_rounds = int(cfg.get("min_rounds", 0) or 0)
    max_rounds = int(cfg.get("rounds", 0) or 0)
    patience = int(cfg.get("patience", 0) or 0)
    labels = dict(METHODS)
    want = list(methods) if methods else [m[0] for m in METHODS]
    shown = [v for v in want if curves_of(runs, v)]
    missing = [v for v in want if v not in shown]
    if missing:
        print(f"no validation curves for: {', '.join(missing)} (left out)", flush=True)
    if not shown:
        print("none of the methods has validation curves in train_log.json: no convergence figure", flush=True)
        return []

    rows, rng, ends = [], np.random.default_rng(0), {}
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS,
                         "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
                         "ytick.labelcolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK}):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
        # grey methods first, so the two coloured ones are drawn on top
        order = [v for v in shown if v not in HIGHLIGHT] + [v for v in ("fedavg", "dtg2") if v in shown]
        n_runs = {}
        for v in order:
            C = curves_of(runs, v)
            n_runs[v] = len(C)
            per = local_epochs if v == "central" else 1      # centralized: one evaluation per epoch
            x, med, q1, q3 = per_round_stats(C)
            x = x / per
            colour = HIGHLIGHT.get(v)
            if v == "dtg2":
                ax.fill_between(x, q1, q3, color=colour, alpha=0.10, lw=0, zorder=1)
            ax.plot(x, med, color=colour or GREY_LINE, lw=1.8 if colour else 0.9, zorder=4 if colour else 2,
                    solid_joinstyle="round", solid_capstyle="round")
            if colour:
                ends[v] = (x[-1], med[-1])
        for j, v in enumerate(shown):                       # panel (b): one row per method, first on top
            C = curves_of(runs, v)
            per = local_epochs if v == "central" else 1
            best = np.array([(np.nanargmin(c) + 1) / per for c in C if np.isfinite(c).any()])
            ran = np.array([len(c) / per for c in C])
            y = len(shown) - 1 - j
            colour = HIGHLIGHT.get(v)
            bx.scatter(best, y + rng.uniform(-0.22, 0.22, len(best)), s=7, color=colour or GREY_DOT,
                       alpha=0.55 if colour else 0.45, lw=0, zorder=2)
            b1, bm, b3 = np.percentile(best, [25, 50, 75])
            bx.plot([b1, b3], [y, y], color=INK, lw=1.4, zorder=3, solid_capstyle="butt")
            bx.plot(bm, y, marker="|", color=INK, ms=10, mew=1.8, zorder=4)
            still = int(np.sum((ran >= max_rounds) & (best > max_rounds - patience))) if max_rounds and patience else 0
            rows.append({"method": labels.get(v, v), "variant": v, "runs": len(C), "best_round_median": float(bm),
                         "best_round_q1": float(b1), "best_round_q3": float(b3),
                         "rounds_run_median": float(np.median(ran)), "rounds_run_max": float(ran.max()),
                         "runs_still_improving_at_max": still})
        xmax = max(max(e[0] for e in ends.values()) if ends else 0, ax.get_xlim()[1])
        ax.set_xlim(0.5, xmax + 0.18 * xmax)               # room for the end labels
        lo, hi = ax.get_ylim()
        if len(ends) == 2:                                  # direct labels only when the line ends separate
            (_, ya), (_, yb) = ends.values()
            separate = abs(ya - yb) > 0.06 * (hi - lo)
        else:
            separate = bool(ends)
        if separate:
            for v, (xe, ye) in ends.items():
                ax.annotate("DTG-v2" if v == "dtg2" else "FedAvg", (xe, ye), xytext=(4, 0),
                            textcoords="offset points", va="center", ha="left", fontsize=7.5, color=INK2)
        if min_rounds:
            ax.axvline(min_rounds, color=AXIS, lw=0.8, ls=":", zorder=1)
            ax.text(min_rounds, 0.02, f"minimum {min_rounds} rounds", transform=ax.get_xaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color=INK2, zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.85})
        n = sorted(set(n_runs.values()))
        ax.set(xlabel="Communication round", ylabel="Validation loss",
               title=f"(a) Median of {n[0]} runs per round" if len(n) == 1 else "(a) Median over runs per round")
        round_ticks(ax, xmax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        bx.set_yticks(range(len(shown)))
        bx.set_yticklabels([labels.get(v, v) for v in reversed(shown)], color=INK)
        bx.set_ylim(-0.6, len(shown) - 0.4)
        bx.set_xlim(left=0)
        round_ticks(bx, bx.get_xlim()[1])
        bx.set(xlabel="Round with the lowest validation loss", title="(b) Model kept by early stopping")
        bx.grid(axis="x", color=GRID, lw=0.6)
        for axis in (ax, bx):
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)
        handles = [Line2D([], [], color=HIGHLIGHT["dtg2"], lw=1.8), Patch(color=HIGHLIGHT["dtg2"], alpha=0.10, lw=0),
                   Line2D([], [], color=HIGHLIGHT["fedavg"], lw=1.8), Line2D([], [], color=GREY_LINE, lw=0.9)]
        names = ["DTG-v2 (proposed), median", "DTG-v2, interquartile range", "FedAvg, median",
                 "Other methods"]
        fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01),
                   handlelength=1.8, columnspacing=1.6, labelcolor=INK)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    with open(os.path.join(out, "convergence_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"settings: at most {max_rounds} rounds, at least {min_rounds}, patience {patience}, "
          f"local epochs {local_epochs}")
    print(f"{'method':<26}{'runs':>5}  {'best round: median [IQR]':<27}{'rounds run: median (max)':<27}"
          "runs still improving at the maximum")
    for r in rows:
        best = f"{r['best_round_median']:.1f} [{r['best_round_q1']:.1f}, {r['best_round_q3']:.1f}]"
        ran = f"{r['rounds_run_median']:.1f} ({r['rounds_run_max']:.0f})"
        print(f"{r['method']:<26}{r['runs']:>5}  {best:<27}{ran:<27}{r['runs_still_improving_at_max']}")
    print(f"written: {os.path.join(out, name + '.png')} (.pdf), convergence_summary.csv", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loo", required=True, help="leave-one-tube-out results folder (with train_log.json)")
    ap.add_argument("--out", default="", help="default: <loo>/../figures")
    ap.add_argument("--methods", default=",".join(m[0] for m in METHODS),
                    help="variants to show, in this order (default: the federated methods and references)")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.loo)), "figures")
    if not make(a.loo, out, [v.strip() for v in a.methods.split(",") if v.strip()]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
