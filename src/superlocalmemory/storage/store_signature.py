# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Has anything been committed to memory.db since a value was computed?

Recall re-ran the same ``COUNT(*)`` queries on every call to learn whether its
cached entity graph was still current: on a 22k-fact / 481k-edge store that
was 44 ms of each recall, to learn nothing had changed. A commit always writes
the store: in WAL mode the ``-wal`` file (a checkpoint then rewrites the main
file), otherwise the main file itself. So the identity, size and nanosecond
modification time of both files, read BEFORE the value is computed, prove the
value still current when they are unchanged. A commit landing between the
read and the computation changes the signature, so the value is recomputed
next time: it can be recomputed needlessly, never reused stale. A replaced or
restored file has a new identity.

A path that cannot be read gives ``None``, which never matches: no reuse.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

Signature = tuple


def _stat(path: str) -> tuple[int, int, int, int] | None:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return (0, 0, 0, 0)  # no WAL file yet is a state too
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def of(db_path: str | Path | None) -> Signature | None:
    """The store's commit signature, or None when it cannot be read."""
    if db_path is None:
        return None
    main = str(db_path)
    first = _stat(main)
    if first is None or first == (0, 0, 0, 0):
        return None
    wal = _stat(main + "-wal")
    if wal is None:
        return None
    return (first, wal)


def of_db(db: Any) -> Signature | None:
    """``of`` for a DatabaseManager (its ``db_path``)."""
    return of(getattr(db, "db_path", None))


__all__ = ["Signature", "of", "of_db"]
