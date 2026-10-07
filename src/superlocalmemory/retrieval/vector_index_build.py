# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Decode a profile's stored vectors for the in-memory index in a child process.

WHY
---
Most stored embeddings are JSON text. Building the in-memory vector index
decodes every one of them -- on a copy of a 22k-fact store, about 2 s of pure
interpreter time on the daemon's warm-up thread, just as the first recalls
arrive. Sampled: a recall 7.5 s after the daemon became ready took 3.7 s while
its database reads waited on the interpreter lock behind that decoding.

WHAT THIS DOES
--------------
A spawned low-priority child (core/background_process) reads the same rows
with the same predicate and decodes them with the same function
(``canonical_vector_index.partition_arrays``); the daemon receives the ids and
the float32 matrix as bytes. The freshness rule is untouched: the cache read the
change-log head before asking, and the child reads the rows after, so a write
in between is applied again on the next read, never skipped
(retrieval/change_tracked.py). If the child cannot run, the caller builds in
place exactly as before.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 300.0


def child_arrays(db_path: str, profile_id: str, dim: int) -> tuple[list[str], bytes]:
    """Runs in the child: ``(ids, matrix bytes)`` for the profile's visible facts."""
    from superlocalmemory.retrieval.canonical_vector_index import partition_arrays
    from superlocalmemory.storage.database import visible_fact_clause_for_connection

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    try:
        conn.execute("PRAGMA query_only=ON")
        clause = visible_fact_clause_for_connection(conn)
        rows = conn.execute(
            "SELECT fact_id, embedding FROM atomic_facts WHERE profile_id = ?"
            f"{clause} ORDER BY fact_id", (profile_id,))
        ids, matrix = partition_arrays(rows, dim)
    finally:
        conn.close()
    return ids, matrix.astype(np.float32, copy=False).tobytes()


def arrays_in_child(db: Any, profile_id: str, dim: int) -> tuple[list[str], np.ndarray] | None:
    """The child's ``(ids, matrix)``, or None to build in place."""
    db_path = getattr(db, "db_path", None)
    if db_path is None:
        return None
    from superlocalmemory.core.background_process import run_in_child

    try:
        ids, raw = run_in_child(child_arrays, str(db_path), profile_id, int(dim),
                                timeout=TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 -- the in-place build is the fallback
        logger.warning("vector index decode in a separate process failed (%s); "
                       "building in place", type(exc).__name__)
        return None
    matrix = np.frombuffer(bytearray(raw), dtype=np.float32).reshape(len(ids), dim)
    return ids, matrix


__all__ = ["arrays_in_child", "child_arrays"]
