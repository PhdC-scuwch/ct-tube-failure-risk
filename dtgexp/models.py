"""Normalization layer used by the multi-branch network (dtgexp/failrisk.py)."""
from __future__ import annotations

import torch.nn as nn


def norm1d(kind: str, c: int) -> nn.Module:
    """'bn': BatchNorm (default); 'gn': GroupNorm; 'none': identity."""
    if kind == "bn":
        return nn.BatchNorm1d(c)
    if kind == "gn":
        return nn.GroupNorm(num_groups=min(8, c), num_channels=c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)
