# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Run one pure computation in a spawned, low-priority child process.

For work that is all interpreter time -- decoding thousands of stored vectors,
graph algorithms over hundreds of thousands of edges. On the daemon's own
interpreter such work holds the interpreter lock in long stretches, and every
recall thread then waits up to the switch interval each time one of its
database reads hands the lock back (measured: 60 facts by id 15 -> 518 ms
beside one busy thread). A child process shares nothing with the recalls.

The function and its arguments must be importable and picklable; the child
starts fresh (``spawn``), so it never inherits the daemon's threads or locks.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from collections.abc import Callable
from typing import Any

NICE_INCREMENT = 10


def _lower_priority() -> None:
    try:
        os.nice(NICE_INCREMENT)
    except (AttributeError, OSError):  # Windows has no nice; priority is best effort
        pass


def run_in_child(fn: Callable[..., Any], *args: Any, timeout: float) -> Any:
    """``fn(*args)`` in a fresh child process. Raises what it raised, or TimeoutError."""
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        initializer=_lower_priority)
    try:
        return pool.submit(fn, *args).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


__all__ = ["NICE_INCREMENT", "run_in_child"]
