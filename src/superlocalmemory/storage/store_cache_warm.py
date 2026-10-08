# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Read the store once when the daemon starts, so the first recalls find it in memory.

WHY
---
Right after a start the operating system has none of the store in its file
cache, and a recall's reads land on rows scattered across the file. Measured
on a copy of a 2 GB, 22,175-fact store: the first lookup of a popular entity
(about 1,800 facts) took 0.5-3 s and the 50-newest query 5.8 s; once cached,
11 ms and 0.1 ms. Reading the whole file in order took 354 ms, after which the
same lookup took 11 ms. Those cold reads were the recalls over 3 s in the
first minute after a start.

WHAT THIS DOES
--------------
One background thread reads ``memory.db`` and its WAL front to back and drops
the bytes. Nothing is written or changed, so no answer can change. The read
releases the interpreter lock while it waits on the disk. A store larger than
a quarter of this computer's memory is not read: the cache could not keep it,
and it would push out what other programs need. Unknown memory size: nothing
is read.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from superlocalmemory.core.machine import total_ram_gb

logger = logging.getLogger(__name__)

CHUNK = 8 << 20
MAX_FRACTION_OF_RAM = 0.25


def warm(paths: Iterable[str | Path], *, max_bytes: int) -> int:
    """Read each existing file in full, in order, while the total stays within
    ``max_bytes``. Returns the bytes read. Never raises."""
    total = 0
    buf = bytearray(CHUNK)
    for path in paths:
        try:
            size = os.path.getsize(path)
            if total + size > max_bytes:
                continue
            with open(path, "rb", buffering=0) as f:
                while n := f.readinto(buf):
                    total += n
        except OSError:
            continue
    return total


def warm_engine_store(engine: Any) -> int:
    """Read the engine's store and WAL, capped by this computer's memory."""
    db_path = getattr(getattr(engine, "_db", None), "db_path", None)
    ram_gb = total_ram_gb()
    if db_path is None or ram_gb <= 0:
        return 0
    started = time.monotonic()
    read = warm([db_path, f"{db_path}-wal"],
                max_bytes=int(ram_gb * (1 << 30) * MAX_FRACTION_OF_RAM))
    logger.info("store read into the file cache: %d MB in %.0f ms",
                read >> 20, (time.monotonic() - started) * 1000.0)
    return read


def start(engine: Any) -> threading.Thread:
    """Run ``warm_engine_store`` on a daemon thread; returns the started thread."""
    def _run() -> None:
        try:
            warm_engine_store(engine)
        except Exception as exc:  # noqa: BLE001 -- costs speed only, never answers
            logger.debug("store cache warm failed: %s", exc)

    thread = threading.Thread(target=_run, daemon=True, name="store-cache-warm")
    thread.start()
    return thread


__all__ = ["CHUNK", "MAX_FRACTION_OF_RAM", "start", "warm", "warm_engine_store"]
