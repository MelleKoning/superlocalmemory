# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Vectors that no memory can be reached through any more.

Meaning search joins each nearest-neighbour hit to ``embedding_metadata``. A
vec0 row with no metadata and no ``vector_row_map`` row can never become a
result, yet it still takes one of the k slots every search asks for, so the
search has to widen to find real ones. ``VectorStore.gc_orphaned_vectors`` was
written for exactly this and never called; a real 22k-fact store had 809.

This module owns the sweep so both callers share one definition: the routine
maintenance pass (``sweep_unreferenced_vectors``) and the store repair
(``unreferenced_rowids`` + ``fetch_vectors`` + ``delete_rowids``, which keeps an
undo copy). Every write runs under the process write lock in a short
``BEGIN IMMEDIATE`` transaction, the same discipline every vec0 writer uses, so
a vector written together with its references can never be swept in between.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

#: Rows removed per transaction; keeps each write-lock hold short.
BATCH = 200


@contextmanager
def vec_connection(db_path: str | Path) -> Iterator[sqlite3.Connection | None]:
    """A connection with the vector extension loaded, or None if it cannot be."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:  # no extension on this Python: nothing to sweep
            logger.debug("vector extension unavailable: %s", exc)
            yield None
            return
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='fact_embeddings'").fetchone()
        yield conn if has_table else None
    finally:
        conn.close()


def _referenced(conn: sqlite3.Connection) -> set[int]:
    """Every vec0 rowid a fact can be reached through. Read once per call:
    ``vector_row_map`` has no index on ``vec_rowid``, so a per-row NOT EXISTS
    would be quadratic on a large store."""
    refs: set[int] = set()
    for table in ("embedding_metadata", "vector_row_map"):
        try:
            refs.update(int(r[0]) for r in conn.execute(f"SELECT vec_rowid FROM {table}"))  # noqa: S608
        except sqlite3.OperationalError as exc:  # an old store without the table
            if "no such table" not in str(exc).lower():
                raise
    return refs


def unreferenced_rowids(conn: sqlite3.Connection, limit: int | None = None) -> list[int]:
    """vec0 rowids named by neither ``embedding_metadata`` nor ``vector_row_map``."""
    refs = _referenced(conn)
    out = sorted(int(r[0]) for r in conn.execute("SELECT rowid FROM fact_embeddings")
                 if int(r[0]) not in refs)
    return out if limit is None else out[: int(limit)]


def fetch_vectors(conn: sqlite3.Connection, rowids: list[int]) -> list[tuple[int, str, bytes]]:
    """``(rowid, profile_id, embedding bytes)`` for an undo copy."""
    out = []
    for rowid in rowids:
        row = conn.execute(
            "SELECT rowid, profile_id, embedding FROM fact_embeddings WHERE rowid = ?",
            (rowid,)).fetchone()
        if row is not None:
            out.append((int(row[0]), str(row[1]), bytes(row[2])))
    return out


def delete_rowids(conn: sqlite3.Connection, db_path: str | Path, rowids: list[int]) -> int:
    """Delete the given vec0 rows, re-checking each is still unreferenced."""
    from superlocalmemory.storage.write_lock import get_write_lock

    removed = 0
    with get_write_lock(Path(db_path)):
        conn.execute("BEGIN IMMEDIATE")
        try:
            refs = _referenced(conn)
            for rowid in rowids:
                if rowid not in refs:
                    cur = conn.execute("DELETE FROM fact_embeddings WHERE rowid = ?", (rowid,))
                    removed += max(0, cur.rowcount)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return removed


def sweep_unreferenced_vectors(db_path: str | Path, *, limit: int = BATCH) -> int:
    """Maintenance: remove at most ``limit`` unreachable vectors. Never raises."""
    try:
        with vec_connection(db_path) as conn:
            if conn is None:
                return 0
            rowids = unreferenced_rowids(conn, limit)
            return delete_rowids(conn, db_path, rowids) if rowids else 0
    except Exception as exc:
        logger.debug("unreferenced vector sweep skipped: %s", exc)
        return 0


__all__ = ["BATCH", "delete_rowids", "fetch_vectors", "sweep_unreferenced_vectors",
           "unreferenced_rowids", "vec_connection"]
