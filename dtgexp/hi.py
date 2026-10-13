"""Protocol-conditioned filament health index (HI) from Siemens tube-history logs.

Why: the filament current depends strongly on the requested tube current, voltage and focal spot (for example,
filament_begin is about 906 at 20 mA, 1192 at 400 mA and 1255 at 650 mA), and hospitals change their protocols
over the years, while the degradation signal at a FIXED operating point is only about 1-2 % over a whole life.
The raw filament current therefore tracks the protocol rather than the tube.

sliced_filament_ratio: relative filament current at fixed operating points, causal and chain-linked:
  * keep stable scans only (dom_type == 'No', 0 < scan_time_act <= 1.2 s, xc_drops != 0); the same rule for
    every log, whether or not it records the scan type;
  * slice = identical (voltage_nom, current_nom, focus);
  * a slice's baseline is fixed from its first `base_n` scans; a slice that first appears later (protocol
    change) is anchored to the merged level at that moment, so that new slices do not pull the trend back to 1;
  * r(t) = rolling median of the merged ratios over the last `window` stable scans, forward-filled.
r(t) = 1 at the start of the log and uses no future data. In every fold, HI = 1 - (1 - r) / D, where D is the
median relative decline reached at the end of life by the TRAINING tubes (dtgexp/failrisk.py: eol_decline).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd


def sliced_filament_ratio(df: pd.DataFrame, fil_col: str = "filament_begin",
                          slice_cols: Sequence[str] = ("voltage_nom", "current_nom", "focus"),
                          base_n: int = 50, window: int = 500, max_scan_time: float = 1.2,
                          scan_time_col: str = "scan_time_act", dom_col: str = "dom_type",
                          dom_value: str = "No"):
    n = len(df)
    fil = pd.to_numeric(df[fil_col], errors="coerce").to_numpy(float, copy=True)
    ok = np.isfinite(fil) & (fil > 0)
    if dom_col and dom_col in df.columns:
        ok &= df[dom_col].astype(str).str.strip().str.lower().eq(dom_value.lower()).to_numpy()
    if scan_time_col in df.columns and max_scan_time:
        st = pd.to_numeric(df[scan_time_col], errors="coerce").to_numpy(float)
        ok &= (st > 0) & (st <= max_scan_time)
    if "xc_drops" in df.columns:          # exposure command: 0 = no exposure (self-check, aborted scan)
        ok &= ~(pd.to_numeric(df["xc_drops"], errors="coerce").to_numpy(float) == 0)
    # NOTE: scan type ('kind') is deliberately NOT used: only some logs contain it, and the daily
    # service/calibration rotations (ROT) are the most standardized filament measurements available.
    cols = [c for c in slice_cols if c in df.columns]
    keys = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").round(0) for c in cols})
    ok &= keys.notna().all(axis=1).to_numpy()
    key = pd.util.hash_pandas_object(keys.fillna(-1), index=False).to_numpy()
    pos = np.where(ok)[0]
    ratios = np.full(len(pos), np.nan)
    baseline, pending, hist = {}, defaultdict(list), []
    for j, p in enumerate(pos):
        k = key[p]
        b = baseline.get(k)
        if b is not None:
            r = fil[p] / b
            ratios[j] = r
            hist.append(r)
            continue
        pend = pending[k]
        pend.append(fil[p])
        if len(pend) >= base_n:
            level = float(np.median(hist[-window:])) if len(hist) >= base_n else 1.0
            baseline[k] = float(np.median(pend)) / level
            del pending[k]
    r_sel = pd.Series(ratios).rolling(window, min_periods=max(10, window // 5)).median().to_numpy()
    out = np.full(n, np.nan)
    out[pos] = r_sel
    out = pd.Series(out).ffill().to_numpy()
    return out, {"stable_scans": int(ok.sum()), "slices": len(baseline), "slice_cols": cols}
