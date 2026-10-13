#!/usr/bin/env python3
"""Convert Siemens CT tube-history logs (tube_history_<serial>.tua[.gz]) into one CSV per tube (the 37 log fields
used by the feature code + 'kind'), and write an objective per-tube record table.

  python scripts/convert_tua.py logs/scanner1.zip --out data/tubes

Inputs may be single files, folders or .zip archives (tube_history_*.tua, *.tua.gz or *.gz).
Output per tube: device_T<tube serial>.csv, rows sorted by scan count; tubes still in service (no
de-installation date, or 'current' in the file name) go to <out>/in_service/ because they have no
failure point. Further fields are appended at the end (mode, reason, tz_offset_min, system_serial,
tube_serial); use --compat to leave them out.

tube_records.csv / printed table, per tube (facts from the log, independent of any model):
  installation / de-installation date (header), scans, first scan count (does the log start at
  installation?), coverage, largest gap, whether the tube is still in service, the share of scans
  aborted almost immediately (act. scan time < 20% of nominal) over the whole life and in the last
  N scans together with the most frequent abort reasons there (end-of-life signature), and - with
  --old-dir - whether the tube is already present among previously converted CSVs (duplicate check).
"""
import argparse
import glob
import gzip
import io
import os
import sys
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtgexp.cli import setup_console  # noqa: E402
from dtgexp.ct_data import read_csv_any  # noqa: E402

RENAME = {
    "Scan count": "scan_count", "kind": "kind", "date/time": "datetime", "rot time": "rot_time",
    "nom. scan time": "scan_time_nom", "act. scan time": "scan_time_act", "nom. voltage": "voltage_nom",
    "act. voltage": "voltage_act", "nom. current": "current_nom", "min. current": "current_min",
    "max. current": "current_max", "mean current": "current_mean", "begin current": "current_begin",
    "end current": "current_end", "focus": "focus", "frequency": "anode_freq",
    "nom. fil. curr.": "filament_nom", "act. fil. curr. begin": "filament_begin",
    "act. fil. curr. end": "filament_end", "nom. dose": "dose_nom", "min. dose": "dose_min",
    "max. dose": "dose_max", "end dose": "dose_end", "oil temp. begin": "oil_temp_begin",
    "oil temp. end": "oil_temp_end", "calc. temp. anode": "calc_temp_anode",
    "calc. temp. anode cooling interface": "calc_temp_anode_cooling_interface",
    "oil pressure": "oil_pressure", "gantry temp.": "gantry_temp", "calc. temp. oil": "calc_temp_oil",
    "motor temp.": "motor_temp", "nom. current wo. sigma corr.": "nom_current_wo_sigma_corr",
    "motor curr.": "motor_curr", "arcings one": "arcings_one", "xc drops": "xc_drops",
    "arcings both": "arcings_both", "region": "region", "DOM type": "dom_type", "mode": "mode",
    "reason": "reason", "TimeDiffToUTC_Time": "tz_offset_min", "system ser. no.": "system_serial",
    "tube ser. no.": "tube_serial"}
OLD_ORDER = ["datetime", "scan_count", "kind", "rot_time", "scan_time_nom", "scan_time_act", "voltage_nom",
             "voltage_act", "current_nom", "current_min", "current_max", "current_mean", "current_begin",
             "current_end", "focus", "anode_freq", "filament_nom", "filament_begin", "filament_end",
             "dose_nom", "dose_min", "dose_max", "dose_end", "oil_temp_begin", "oil_temp_end",
             "oil_pressure", "gantry_temp", "calc_temp_oil", "motor_temp", "nom_current_wo_sigma_corr",
             "motor_curr", "arcings_one", "xc_drops", "arcings_both", "region", "dom_type",
             "calc_temp_anode", "calc_temp_anode_cooling_interface"]
EXTRA = ["mode", "reason", "tz_offset_min", "system_serial", "tube_serial"]
KEY = ["scan_count", "current_nom", "filament_begin", "filament_end", "voltage_act"]


def is_tube_log(name: str) -> bool:
    """tube_history_<serial>.tua / .tua.gz / .gz (both naming variants seen), or anything with '.tua'."""
    b = os.path.basename(name).lower()
    return (".tua" in b or b.endswith("_tua.gz")
            or (b.startswith("tube_history") and (b.endswith(".gz") or b.endswith(".tua") or b.endswith(".txt"))))


def iter_sources(paths):
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "**", "*"), recursive=True)):
                if os.path.isfile(f) and is_tube_log(f):
                    yield os.path.basename(f), lambda f=f: open(f, "rb").read()
        elif p.lower().endswith(".zip"):
            z = zipfile.ZipFile(p)
            for m in sorted(z.namelist()):
                if is_tube_log(m):
                    yield os.path.basename(m), lambda z=z, m=m: z.read(m)
        else:
            yield os.path.basename(p), lambda p=p: open(p, "rb").read()


def parse_tua(raw: bytes):
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("#Scan count")), None)
    if hdr is None:
        raise ValueError("no '#Scan count' column header found")
    meta = {}
    for l in lines[:hdr]:
        if l.startswith("#") and ":" in l:
            k, v = l[1:].split(":", 1)
            meta[k.strip()] = v.strip()
    cols = lines[hdr].lstrip("#").split("\t")
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr + 1:])), sep="\t", header=None, names=cols,
                     keep_default_na=False, na_values=[""], low_memory=False)
    df = df.rename(columns=RENAME)
    for c in df.columns:
        if c not in ("kind", "datetime", "region", "dom_type", "mode", "reason"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["scan_count"]).sort_values("scan_count", kind="stable")
    df = df.drop_duplicates(subset=["scan_count"], keep="first").reset_index(drop=True)
    return meta, df


def key_set(df, n=None):
    k = df[[c for c in KEY if c in df.columns]].apply(pd.to_numeric, errors="coerce").round(1)
    if n and len(k) > n:
        k = k.iloc[np.linspace(0, len(k) - 1, n).astype(int)]
    return set(map(tuple, k.dropna().to_numpy()))


def record(name, meta, df, window):
    t = pd.to_datetime(df["datetime"], errors="coerce")
    inst = pd.to_datetime(meta.get("Date of Installation"), errors="coerce")
    deinst = pd.to_datetime(meta.get("Date of Deinstallation"), errors="coerce")
    first, last = int(df["scan_count"].iloc[0]), int(df["scan_count"].iloc[-1])
    gaps = t.diff().dt.total_seconds().div(86400).fillna(0)
    nom = pd.to_numeric(df["scan_time_nom"], errors="coerce")
    act = pd.to_numeric(df["scan_time_act"], errors="coerce")
    aborted = ((nom >= 0.5) & (act < 0.2 * nom)).to_numpy()
    tail = slice(max(0, len(df) - window), len(df))
    reasons = df["reason"].astype(str).str.strip() if "reason" in df.columns else pd.Series([""] * len(df))
    tail_reasons = Counter(r for r, a in zip(reasons.iloc[tail], aborted[tail]) if a and r not in ("", "nan"))
    in_service = pd.isna(deinst) or "current" in name.lower()
    return {
        "file": name, "tube_serial": meta.get("Tube Serial Number", ""), "system_serial": meta.get("System Serial Number", ""),
        "tube_type": meta.get("Tube Type", ""), "software": meta.get("Software version", ""),
        "installed": str(inst.date()) if pd.notna(inst) else "", "deinstalled": str(deinst.date()) if pd.notna(deinst) else "",
        "in_service": bool(in_service), "scans_header": meta.get("Number of Scans", ""), "rows": len(df),
        "first_scan": first, "last_scan": last, "coverage": round(len(df) / max(1, last - first + 1), 4),
        "first_time": str(t.iloc[0]), "last_time": str(t.iloc[-1]),
        "days_install_to_first": round((t.iloc[0] - inst).total_seconds() / 86400, 1) if pd.notna(inst) else np.nan,
        "days_last_to_deinstall": round((deinst - t.iloc[-1]).total_seconds() / 86400, 1) if pd.notna(deinst) else np.nan,
        "max_gap_days": round(float(gaps.max()), 1), "gaps_over_30d": int((gaps > 30).sum()),
        "abort_share_life_%": round(100 * float(aborted.mean()), 2),
        f"abort_share_last{window}_%": round(100 * float(aborted[tail].mean()), 2),
        "top_abort_reasons_last": "; ".join(f"{r}:{n}" for r, n in tail_reasons.most_common(3)),
    }


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".tua/.tua.gz files, folders or .zip archives")
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-dir", default=None, help="folder with previously converted device_*.csv (duplicate check)")
    ap.add_argument("--window", type=int, default=500, help="last N scans for the end-of-life signature")
    ap.add_argument("--compat", action="store_true", help="write only the old columns (no extra fields)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    old = {}
    if a.old_dir:
        for f in sorted(glob.glob(os.path.join(a.old_dir, "device_*.csv"))):
            head = read_csv_any(f, nrows=3)
            old[os.path.splitext(os.path.basename(f))[0]] = key_set(
                read_csv_any(f, usecols=[c for c in KEY if c in head.columns]), n=400)
    recs = []
    for name, load in iter_sources(a.inputs):
        try:
            meta, df = parse_tua(load())
        except Exception as e:
            print(f"{name}: skipped ({e})")
            continue
        serial = meta.get("Tube Serial Number") or (str(int(df["tube_serial"].dropna().iloc[0]))
                                                    if "tube_serial" in df and df["tube_serial"].notna().any() else "")
        tag = serial if serial else os.path.splitext(name)[0]
        if "current" in name.lower():
            tag += "_current"
        cols = [c for c in OLD_ORDER if c in df.columns] + ([] if a.compat else [c for c in EXTRA if c in df.columns])
        rec = record(name, meta, df, a.window)
        out_name = f"device_T{tag}.csv"
        # tubes still in service have no failure point -> kept apart so the exporter does not treat
        # their last scan as end of life
        sub = os.path.join(a.out, "in_service") if rec["in_service"] else a.out
        os.makedirs(sub, exist_ok=True)
        df[cols].to_csv(os.path.join(sub, out_name), index=False, encoding="utf-8")
        rec["csv"] = os.path.relpath(os.path.join(sub, out_name), a.out)
        if old:
            ks = key_set(df)
            best = max(((sum(k in ks for k in v) / max(1, len(v)), o) for o, v in old.items()), default=(0, ""))
            rec["same_as_old"] = best[1] if best[0] > 0.9 else ""
            rec["old_match_share"] = round(best[0], 3)
        recs.append(rec)
        print(f"{name}: {len(df)} scans -> {out_name}")
    if not recs:
        raise SystemExit("no tube-history files converted")
    R = pd.DataFrame(recs)
    R.to_csv(os.path.join(a.out, "tube_records.csv"), index=False, encoding="utf-8")
    show = ["tube_serial", "system_serial", "installed", "deinstalled", "in_service", "rows", "first_scan",
            "last_scan", "coverage", "max_gap_days", "abort_share_life_%", f"abort_share_last{a.window}_%",
            "top_abort_reasons_last"] + (["same_as_old"] if old else [])
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + R[show].to_string(index=False))
    print(f"\nwritten: {len(R)} tube CSVs and tube_records.csv in {a.out}")


if __name__ == "__main__":
    main()
