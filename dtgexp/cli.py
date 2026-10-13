"""Helpers shared by the scripts: console encoding and automatic device selection."""
from __future__ import annotations

import sys
from typing import Dict


def setup_console():
    """UTF-8 output on Windows consoles and redirected output (tables contain special characters)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def resolve_device(spec: str = "auto") -> str:
    """'auto' -> the first CUDA GPU if PyTorch sees one, otherwise the CPU. An explicit 'cuda...' on a machine
    without a usable GPU falls back to the CPU with a warning, so that one command works on every machine."""
    import torch
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: --device {spec} requested but no CUDA GPU is available; using CPU", file=sys.stderr)
        return "cpu"
    return spec


def hardware_info(device: str) -> Dict[str, str]:
    import platform
    import torch
    name = torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else \
        (platform.processor() or platform.machine() or "CPU")
    return {"device": device, "name": name, "host": platform.node(), "torch": torch.__version__}
