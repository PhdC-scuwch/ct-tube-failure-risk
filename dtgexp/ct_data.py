"""File helpers for the tube logs."""
from __future__ import annotations

import pandas as pd


def read_csv_any(path: str, **kw) -> pd.DataFrame:
    """read_csv that also accepts UTF-8 files with a byte-order mark (e.g. saved by Excel) and GBK-encoded files."""
    last = None
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError as e:
            last = e
    raise last
