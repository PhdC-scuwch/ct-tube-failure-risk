#!/usr/bin/env python3
"""Weibull life distribution of the failed tubes (Sec. IV-A and V-A of the paper), from the tube records written by
scripts/convert_tua.py: maximum-likelihood shape beta and scale eta (in scans) with the 95% profile-likelihood interval
of beta, for the tubes past burn-in and, for comparison, with the infant failures included.

  python scripts/weibull_lives.py data/tubes/tube_records.csv [more tube_records.csv ...] --min-life 5000

The life of a tube is its scan counter at de-installation (column scans_header); tubes still in service are left out.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtgexp.ct_data import read_csv_any  # noqa: E402
from dtgexp.survival import fit_weibull, shape_profile_ci  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", help="tube_records.csv files")
    ap.add_argument("--min-life", type=float, default=5000, help="shorter lives are infant failures (default 5000 scans)")
    a = ap.parse_args()
    R = pd.concat([read_csv_any(f) for f in a.records], ignore_index=True)
    R = R[~R["in_service"].astype(str).str.lower().isin(["true", "1"])]
    R = R.drop_duplicates(subset="tube_serial", keep="last")
    life = pd.to_numeric(R["scans_header"], errors="coerce").dropna().to_numpy(float)
    past = life[life >= a.min_life]
    print(f"failed tubes: {len(life)}; infant failures (< {a.min_life:,.0f} scans): {int((life < a.min_life).sum())}")
    for name, t in (("past burn-in", past), ("all failed tubes", life)):
        if len(t) < 3:
            continue
        beta, eta = fit_weibull(t)
        lo, hi = shape_profile_ci(t)
        print(f"{name}: n = {len(t)}, beta = {beta:.2f} (95% CI {lo:.2f} to {hi:.2f}), eta = {eta:,.0f} scans, "
              f"lives {t.min():,.0f} to {t.max():,.0f} scans, coefficient of variation {t.std() / t.mean():.2f}")


if __name__ == "__main__":
    main()
