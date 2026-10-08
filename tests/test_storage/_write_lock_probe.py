# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Measure the shared memory.db write lock, and wait on it by progress.

A concurrency test that joins its writer threads with a fixed wall-clock
timeout measures the disk, not the design: on a loaded machine every commit
is slower, the fixed timeout runs out while the writers are still taking
turns, and the test reports writes that were merely queued as "not completed".

``TimedWriteLock`` stands in for the registry's RLock (same object for every
writer of the path) and records how long each turn held it. ``wait_for_writers``
then waits for as long as the lock keeps changing hands, and fails only when
nobody has released it for many times the longest turn measured: a hang, not
a slow disk.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from superlocalmemory.storage import write_lock

#: Fail only after this many of the longest measured turns pass with no turn
#: ending. With N writers queued, a fair lock hands every one of them a turn
#: well inside N + 1 turns, so this is far beyond any legitimate wait.
STALL_TURNS = 50
#: Lower bound on that window, for runs whose turns are all very short.
STALL_FLOOR_S = 5.0


class TimedWriteLock:
    """A re-entrant lock that records each outermost hold."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self._meta = threading.Lock()
        self.holds: list[float] = []
        self.last_release = time.monotonic()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        got = self._lock.acquire(blocking, timeout)
        if got:
            depth = getattr(self._local, "depth", 0)
            if depth == 0:
                self._local.started = time.monotonic()
            self._local.depth = depth + 1
        return got

    def release(self) -> None:
        depth = self._local.depth - 1
        self._local.depth = depth
        if depth == 0:
            now = time.monotonic()
            with self._meta:
                self.holds.append(now - self._local.started)
                self.last_release = now
        self._lock.release()

    __enter__ = acquire

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _is_owned(self) -> bool:
        return getattr(self._local, "depth", 0) > 0

    @property
    def longest_hold(self) -> float:
        with self._meta:
            return max(self.holds, default=0.0)


def install_timed_lock(db_path: Path) -> TimedWriteLock:
    """Make ``get_write_lock(db_path)`` return a TimedWriteLock from now on.

    Install it before any DatabaseManager for the path is created: a manager
    keeps the lock object it got at construction.
    """
    probe = TimedWriteLock()
    key = str(Path(db_path).resolve())
    with write_lock._registry_lock:
        write_lock._registry[key] = probe  # type: ignore[assignment]
    return probe


def remove_timed_lock(db_path: Path) -> None:
    key = str(Path(db_path).resolve())
    with write_lock._registry_lock:
        write_lock._registry.pop(key, None)


def wait_for_writers(threads: list[threading.Thread], probe: TimedWriteLock) -> None:
    """Join *threads* while the write lock keeps changing hands.

    Raises AssertionError naming the stall when no turn has ended for
    max(STALL_FLOOR_S, STALL_TURNS x longest measured turn).
    """
    for thread in threads:
        while thread.is_alive():
            thread.join(timeout=0.05)
            if not thread.is_alive():
                break
            window = max(STALL_FLOOR_S, STALL_TURNS * probe.longest_hold)
            idle = time.monotonic() - probe.last_release
            if idle > window:
                raise AssertionError(
                    f"{thread.name} is stuck: no writer released the memory.db "
                    f"write lock for {idle:.1f}s (window {window:.1f}s = "
                    f"{STALL_TURNS} x longest turn {probe.longest_hold:.3f}s, "
                    f"floor {STALL_FLOOR_S}s; {len(probe.holds)} turns so far)"
                )
