# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Prove a stopped daemon left no descendants behind, where there is no
process group to sweep (Windows)."""

from __future__ import annotations

import time


def descendants_of(pid: int) -> list:
    """Every live descendant of ``pid`` (workers, a self-heal download)."""
    import psutil

    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []


def reap_survivors(processes: list, *, grace: float) -> list[tuple[int, str]]:
    """Wait up to ``grace`` seconds for ``processes`` to exit (workers notice
    their parent is gone on a 5-10 s poll), kill any that did not, and return
    ``(pid, command line)`` for each of those."""
    deadline = time.monotonic() + grace
    alive = [p for p in processes if p.is_running()]
    while alive and time.monotonic() < deadline:
        time.sleep(1.0)
        alive = [p for p in alive if p.is_running()]
    survivors = []
    for process in alive:
        try:
            command = " ".join(process.cmdline())[:160]
        except Exception:
            command = "?"
        survivors.append((process.pid, command))
        try:
            process.kill()
        except Exception:
            pass
    return survivors
