# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Waiting on the slow processes of a Laya setup — and knowing when to stop.

A setup step used to wait for its process with no way out but the step's own
time budget: 10 minutes for the package, 30 for the weights. On a network that
quietly blocks the download server, the download made no progress and the
dashboard showed "Downloading the model weights…" for half an hour with every
other control refused. Here every wait can end three ways besides finishing:

* the person chose Cancel (``CANCEL`` is set) — the process is stopped at once;
* the step ran past its budget;
* a download stopped growing for ``stall_s`` — the network is blocking it.

Stdlib only, no SLM imports: core/laya_runtime.py builds on it.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

#: Set by the dashboard's Cancel; cleared when a new setup starts. One install
#: runs at a time (laya_runtime holds a cross-process lock), so one flag is enough.
CANCEL = threading.Event()

KIND_CANCELLED = "cancelled"
KIND_STALLED = "stalled"
KIND_TIMEOUT = "timeout"

#: How long a download may sit without one new byte before it counts as blocked.
DEFAULT_STALL_S = 120.0

_TICK_S = 1.0

Outcome = tuple[bool, str, str]  # (ok, error_kind, detail-for-logs-only)


def stop(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and reap it; never raises."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.communicate(timeout=5)
    except Exception:  # noqa: BLE001 — best effort: the process is already killed
        pass


def run(cmd: list[str], *, timeout_s: float, env: dict[str, str] | None,
        classify: Callable[[str], str]) -> Outcome:
    """Run ``cmd`` to the end, unless cancelled or over budget. Never raises."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env)
    except OSError as exc:
        return False, "other", str(exc)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            # communicate() keeps draining both pipes between timeouts, so a
            # chatty process (pip) can never block on a full pipe.
            out, err = proc.communicate(timeout=_TICK_S)
            break
        except subprocess.TimeoutExpired:
            if CANCEL.is_set():
                stop(proc)
                return False, KIND_CANCELLED, "stopped by the person"
            if time.monotonic() >= deadline:
                stop(proc)
                return False, KIND_TIMEOUT, f"{cmd[0]} ran past {timeout_s:.0f} s"
    if proc.returncode == 0:
        return True, "", ""
    detail = (err or out or "")[-2000:]
    return False, classify(detail), detail


def watch_download(proc: subprocess.Popen, measure: Callable[[], int], *,
                   timeout_s: float, stall_s: float,
                   on_size: Callable[[int], None] | None = None,
                   clock: Callable[[], float] = time.monotonic) -> Outcome | None:
    """Wait for a download process. None when it exited by itself (the caller
    reads its result); otherwise why it was stopped.

    ``measure`` returns the bytes on disk so far. A download whose size has not
    changed for ``stall_s`` is stopped: the server is unreachable or blocked,
    and waiting longer only keeps the person staring at a frozen bar.
    ``clock`` is injectable so the rule can be tested without real waiting.
    """
    start = clock()
    last_size, last_change = measure(), start
    while True:
        try:
            proc.wait(timeout=_TICK_S)
            return None
        except subprocess.TimeoutExpired:
            pass
        now = clock()
        if CANCEL.is_set():
            stop(proc)
            return False, KIND_CANCELLED, "stopped by the person"
        if now - start >= timeout_s:
            stop(proc)
            return False, KIND_TIMEOUT, "download exceeded its time budget"
        size = measure()
        if size != last_size:
            last_size, last_change = size, now
        elif now - last_change >= stall_s:
            stop(proc)
            return False, KIND_STALLED, f"no new bytes for {stall_s:.0f} s"
        if on_size is not None:
            try:
                on_size(size)
            except Exception:  # noqa: BLE001 — a bad UI callback can't abort an install
                pass


def folder_size(path: Path) -> int:
    """Bytes in every file under ``path``; 0 when it does not exist."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


__all__ = [
    "CANCEL",
    "DEFAULT_STALL_S",
    "KIND_CANCELLED",
    "KIND_STALLED",
    "KIND_TIMEOUT",
    "folder_size",
    "run",
    "stop",
    "watch_download",
]
