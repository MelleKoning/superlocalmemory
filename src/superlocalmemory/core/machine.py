# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Facts about this computer that model recommendations depend on."""

from __future__ import annotations

import platform
import subprocess


def total_ram_gb() -> float:
    """Total system memory in GB; 0.0 when it cannot be read."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024 ** 3)
        except Exception:  # noqa: BLE001 — unknown, not fatal
            pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:  # noqa: BLE001 — unknown, not fatal
        pass
    return 0.0


__all__ = ["total_ram_gb"]
