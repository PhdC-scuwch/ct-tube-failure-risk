#!/usr/bin/env python3
"""Data-quality audit of the tube logs and the exclusion list (no model, CPU, a few minutes).

Per tube: share of scans missing from the log (scan-counter jumps), calendar gaps (above all in the last 30
days, where the 7- and 30-day labels live), whether the last record is the end of normal clinical work,
counter / timestamp problems, missing values of the fields the features use (overall and in the last 30
days), fields absent from the log format, where the log starts in the tube's life, and how long it takes
to collect the 30 active reference days.

Exclusion criteria - fixed in this file BEFORE any model result per tube was looked at; they use data
quality only, never the precursor signal or the predictions:
  X1 failure time uncertain: the last day with >= 20 scans is more than 3 days before the last record, or
     the gap before the last record exceeds 3 days
  X2 data gap before failure: a gap longer than 7 days within the last 30 days
  X3 scans missing from the log: logged records < 90 % of the scan-counter span, or one counter jump larger
     than 5 % of the tube's life
  X4 corrupted log: the scan counter decreases at > 0.1 % of consecutive records, or > 1 % of timestamps are
     unreadable, out of order or duplicated
  X5 core fields missing: any of scan count, kV, mA, scan time, oil temperature (begin/end) missing in > 5 % of
     the records overall or in the last 30 days
Reported but not excluded: log starting after 5 % of life (robustness subset), idle days, absent optional
fields (the feature code fills them with constants for every day of that tube).

  python scripts/data_audit.py --raw-dir data/tubes --site-map site_map.csv --out results/audit
"""
import argparse
import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dtgexp.hi import sliced_filament_ratio  # noqa: E402

CORE = ["scan_count", "voltage_act", "current_mean", "scan_time_act", "oil_temp_begin", "oil_temp_end"]
OPTIONAL = ["voltage_nom", "current_nom", "current_min", "current_max", "scan_time_nom", "filament_begin", "gantry_temp",
            "oil_pressure", "calc_temp_anode_cooling_interface", "motor_temp", "anode_freq", "motor_curr", "xc_drops",
            "arcings_one", "arcings_both", "reason"]
ACTIVE_MIN = 20
RULES = {"X1": "failure time uncertain", "X2": "data gap > 7 d in the last 30 d", "X3": "scans missing from the log",
         "X4": "corrupted counter or timestamps", "X5": "core fields missing"}


def audit_tube(path, fe):
    name = os.path.splitext(os.path.basename(path))[0]
    raw = pd.read_csv(path, low_memory=False)
    dt_raw = pd.to_datetime(raw["datetime"], errors="coerce") if "datetime" in raw.columns else pd.Series(pd.NaT, index=raw.index)
    out = {"tube": name, "records_raw": len(raw), "bad_timestamps_pct": 100 * float(dt_raw.isna().mean()),
           "timestamps_out_of_order_pct": 100 * float((dt_raw.diff().dt.total_seconds() < 0).mean()),
           "duplicate_timestamps_pct": 100 * float(dt_raw.duplicated().mean())}
    with redirect_stdout(io.StringIO()):
        df = fe.load_raw_csv(path)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    t = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    sc = pd.to_numeric(df["scan_count"], errors="coerce").to_numpy(float)
    scf = pd.Series(sc).ffill().bfill().to_numpy()
    T = float(t[-1])
    life = float(np.nanmax(sc))
    span = float(np.nanmax(sc) - np.nanmin(sc) + 1)
    jumps = np.diff(scf) - 1
    out.update({"first_date": str(df["datetime"].iloc[0].date()), "last_date": str(df["datetime"].iloc[-1].date()),
                "life_scans": life, "log_start_share_of_life": float(scf[0] / life) if life > 0 else np.nan,
                "calendar_days": T, "scans_logged_share_pct": 100 * len(df) / span if span > 0 else np.nan,
                "largest_counter_jump_share_of_life_pct": 100 * float(max(jumps.max(), 0)) / life if len(jumps) and life > 0 else 0.0,
                "counter_backwards_pct": 100 * float((np.diff(scf) < 0).mean()) if len(scf) > 1 else 0.0})
    day = np.floor(t).astype(int)
    n_days = int(day.max()) + 1
    cnt = np.bincount(day, minlength=n_days)
    gaps = np.diff(t)
    tail30 = t >= T - 30
    g30 = np.diff(t[tail30]) if tail30.sum() > 1 else np.array([30.0])
    busy = np.where(cnt >= ACTIVE_MIN)[0]
    act = np.where(cnt >= ACTIVE_MIN)[0]
    out.update({
        "idle_day_share_pct": 100 * float((cnt == 0).mean()),
        "longest_gap_days": float(gaps.max()) if len(gaps) else 0.0,
        "longest_gap_last30_days": float(g30.max()),
        "gap_before_last_record_days": float(gaps[-1]) if len(gaps) else 0.0,
        "days_from_last_clinical_day_to_end": max(0.0, float(T - (busy[-1] + 1))) if len(busy) else np.nan,
        "active_days_last30": int((cnt[max(0, n_days - 30):] >= ACTIVE_MIN).sum()),
        "scans_per_active_day_median": float(np.median(cnt[cnt >= ACTIVE_MIN])) if len(act) else 0.0,
        "calendar_days_to_30_reference_days": float(act[29] + 1) if len(act) >= 30 else np.nan,
    })
    tail_rec = (dt_raw >= dt_raw.max() - pd.Timedelta(days=30)).to_numpy()
    for c in CORE + OPTIONAL:
        if c not in raw.columns:
            out[f"missing_{c}"] = "absent"
            continue
        if c == "reason":                      # empty = normally completed scan; only presence of the field matters
            out[f"missing_{c}"] = "present"
            continue
        v = pd.to_numeric(raw[c], errors="coerce")
        out[f"missing_{c}"] = f"{100 * v.isna().mean():.1f}% / last30 {100 * v.isna()[tail_rec].mean():.1f}%"
        out[f"_miss_{c}"] = (100 * float(v.isna().mean()), 100 * float(v.isna()[tail_rec].mean()))
    try:
        hi, _ = sliced_filament_ratio(df)
        hi = np.asarray(hi, float)
        out["filament_hi_valid_pct"] = 100 * float(np.isfinite(hi).mean())
    except Exception as e:
        out["filament_hi_valid_pct"] = np.nan
        out["filament_hi_note"] = f"failed: {type(e).__name__}"
    return out, cnt


def exclusion_reasons(r):
    why = []
    lcd = r["days_from_last_clinical_day_to_end"]
    if (not np.isfinite(lcd)) or lcd > 3 or r["gap_before_last_record_days"] > 3:
        why.append(f"X1 {RULES['X1']} (last clinical day {lcd:.1f} d before the end, final gap "
                   f"{r['gap_before_last_record_days']:.1f} d)")
    if r["longest_gap_last30_days"] > 7:
        why.append(f"X2 {RULES['X2']} ({r['longest_gap_last30_days']:.1f} d)")
    if r["scans_logged_share_pct"] < 90 or r["largest_counter_jump_share_of_life_pct"] > 5:
        why.append(f"X3 {RULES['X3']} (logged {r['scans_logged_share_pct']:.1f}% of the counter span, largest jump "
                   f"{r['largest_counter_jump_share_of_life_pct']:.1f}% of life)")
    if r["counter_backwards_pct"] > 0.1 or max(r["bad_timestamps_pct"], r["timestamps_out_of_order_pct"],
                                               r["duplicate_timestamps_pct"]) > 1:
        why.append(f"X4 {RULES['X4']}")
    bad = [c for c in CORE if r.get(f"missing_{c}") == "absent" or max(r.get(f"_miss_{c}", (0, 0))) > 5]
    if bad:
        why.append(f"X5 {RULES['X5']} ({', '.join(bad)})")
    return why


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features"),
                    help="folder containing tube_features.py (default: features/)")
    ap.add_argument("--pattern", default="device_*.csv")
    ap.add_argument("--site-map", default="")
    ap.add_argument("--min-life", type=float, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.abspath(a.src))
    import tube_features as fe
    files = sorted(glob.glob(os.path.join(a.raw_dir, a.pattern)),
                   key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p)])
    hosp = {}
    if a.site_map and os.path.exists(a.site_map):
        sm = pd.read_csv(a.site_map, dtype=str, encoding="utf-8-sig").fillna("")
        sm.columns = [c.strip().lower() for c in sm.columns]
        hosp = {str(r.tube).replace("device_", "").strip(): str(r.hospital).strip() for r in sm.itertuples()}
    rows, counts, ex = [], {}, []
    for p in files:
        r, cnt = audit_tube(p, fe)
        r["hospital"] = hosp.get(r["tube"].replace("device_", ""), "")
        infant = r["life_scans"] < a.min_life
        why = [] if infant else exclusion_reasons(r)
        r["used_by_models"] = not infant
        rows.append(r)
        counts[r["tube"]] = cnt
        ex.append({"tube": r["tube"], "hospital": r["hospital"], "excluded": bool(why),
                   "reasons": "; ".join(why) if why else ("infant failure, not modelled" if infant else "")})
        print(f"{r['tube']}: {r['calendar_days']:.0f} days, logged {r['scans_logged_share_pct']:.1f}% of its scans, idle days "
              f"{r['idle_day_share_pct']:.0f}%, longest gap {r['longest_gap_days']:.1f} d (last 30 d: "
              f"{r['longest_gap_last30_days']:.1f} d), last clinical day {r['days_from_last_clinical_day_to_end']:.1f} d "
              f"before the end, log starts at {100 * r['log_start_share_of_life']:.0f}% of life"
              + ("  -> infant failure" if infant else ("  -> EXCLUDED: " + "; ".join(why) if why else "")))
    D = pd.DataFrame(rows)
    D = D[[c for c in D.columns if not c.startswith("_")]]
    lead = ["tube", "hospital", "used_by_models", "first_date", "last_date", "life_scans", "log_start_share_of_life",
            "calendar_days", "scans_logged_share_pct", "largest_counter_jump_share_of_life_pct", "idle_day_share_pct",
            "longest_gap_days", "longest_gap_last30_days", "gap_before_last_record_days",
            "days_from_last_clinical_day_to_end", "active_days_last30", "scans_per_active_day_median",
            "calendar_days_to_30_reference_days", "filament_hi_valid_pct", "counter_backwards_pct",
            "timestamps_out_of_order_pct", "duplicate_timestamps_pct", "bad_timestamps_pct", "records_raw"]
    D = D[[c for c in lead if c in D.columns] + [c for c in D.columns if c not in lead]]
    D.to_csv(os.path.join(a.out, "data_audit.csv"), index=False)
    E = pd.DataFrame(ex)
    E.to_csv(os.path.join(a.out, "exclusion.csv"), index=False)
    used = D[D.used_by_models]
    absent = {c: int((used[f"missing_{c}"] == "absent").sum()) for c in OPTIONAL if f"missing_{c}" in used}
    late = used[used.log_start_share_of_life > 0.05].tube.tolist()
    lines = [f"tubes modelled: {len(used)}; excluded by the data-quality criteria: {int(E.excluded.sum())}",
             *[f"  {r.tube}: {r.reasons}" for r in E.itertuples() if r.excluded],
             f"logs starting after 5% of life (robustness subset leaves them out): {len(late)} - {', '.join(late)}",
             "fields absent from some logs (count of modelled tubes): " +
             (", ".join(f"{k} {v}" for k, v in absent.items() if v) or "none"),
             f"median share of scans present in the logs: {used.scans_logged_share_pct.median():.1f}% "
             f"(lowest {used.scans_logged_share_pct.min():.1f}%)",
             f"median idle-day share: {used.idle_day_share_pct.median():.0f}% (highest {used.idle_day_share_pct.max():.0f}%)"]
    open(os.path.join(a.out, "audit_summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 40):
        open(os.path.join(a.out, "data_audit.txt"), "w", encoding="utf-8").write(D.round(3).to_string(index=False) + "\n")
    names = used.tube.tolist()
    W = 180
    M = np.full((len(names), W), np.nan)
    for i, n in enumerate(names):
        c = counts[n][-W:].astype(float)
        M[i, W - len(c):] = c
    fig, ax = plt.subplots(figsize=(11, 0.35 * max(len(names), 1) + 1.5))
    Mm = np.ma.masked_where(~np.isfinite(M) | (M == 0), np.log10(np.where(M > 0, M, 1)))
    im = ax.imshow(Mm, aspect="auto", cmap="viridis", interpolation="nearest", extent=[-W, 0, len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("device_", "") + (" *" if bool(E.set_index("tube").loc[n, "excluded"]) else "")
                        for n in names], fontsize=7)
    ax.axvline(-30, color="red", lw=0.8, ls="--")
    ax.axvline(-7, color="red", lw=0.8)
    ax.set(xlabel="Days before the last record (dashed: 30 days, solid: 7 days; white: no scans that day; *: excluded)",
           title="Daily scans before the last record (log10)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="log10 scans per day")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig_data_coverage_last180d.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwritten to {a.out}: data_audit.csv, exclusion.csv, audit_summary.txt, data_audit.txt, fig_data_coverage_last180d.png")


if __name__ == "__main__":
    main()
