# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair``: a memory a delete started on and never finished.

A delete writes the memory's erasure tombstone and removes its search entries
before its last protection check and the final row delete. If the writer then
times out or the process dies, the memory stays stored with the tombstone:
it cannot be found, and corrections naming it are refused as "being deleted".
The person was told the delete did not happen.

What the repair does, per stored memory with a tombstone older than
``delete_refusal.STALE_AFTER_S`` (younger ones may belong to a delete still
running):

* **an unfinished ordinary delete** (the erasure's own obligations name this
  memory, or its receipt says ``fact``) — restored, as the person was told:
  the tombstone and its open erase obligations go and every search entry is
  rebuilt from the stored memory (``delete_refusal.heal_stale``). That needs the
  running engine, so without one (SLM not running) it is only counted.
* **an unfinished erasure of a person or a profile** (GDPR), or a delete whose
  origin cannot be proven — never restored: bringing back what someone asked
  to have erased is the one wrong answer. Counted, so the plan says to run the
  erasure again.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

DELETE = "delete"
ERASURE = "erasure"


@dataclass(frozen=True, slots=True)
class Unfinished:
    profile_id: str
    fact_id: str
    erasure_id: str
    origin: str  # DELETE | ERASURE


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                        (table,)).fetchone() is not None


def _is_ordinary_delete(conn: sqlite3.Connection, erasure_id: str, fact_id: str) -> bool:
    if _has(conn, "projection_obligations") and conn.execute(
            "SELECT 1 FROM projection_obligations WHERE operation_id = ? AND kind = 'erase' "
            "AND subject_id = ? LIMIT 1", (erasure_id, fact_id)).fetchone():
        return True
    return _has(conn, "erasure_receipts") and conn.execute(
        "SELECT 1 FROM erasure_receipts WHERE erasure_id = ? AND subject_type = 'fact' "
        "AND subject_id = ? LIMIT 1", (erasure_id, fact_id)).fetchone() is not None


def find(conn: sqlite3.Connection, *, now: float | None = None) -> list[Unfinished]:
    """Every stored memory carrying a tombstone older than the grace period."""
    from superlocalmemory.core.delete_refusal import STALE_AFTER_S

    if not _has(conn, "projection_tombstones"):
        return []
    cutoff = (time.time() if now is None else now) - STALE_AFTER_S
    rows = conn.execute(
        "SELECT t.profile_id, t.fact_id, t.erasure_id FROM projection_tombstones AS t "
        "WHERE t.created_at < ? AND EXISTS (SELECT 1 FROM atomic_facts AS f "
        "WHERE f.fact_id = t.fact_id AND f.profile_id = t.profile_id) "
        "ORDER BY t.created_at, t.fact_id", (cutoff,)).fetchall()
    return [Unfinished(str(p), str(f), str(e),
                       DELETE if _is_ordinary_delete(conn, str(e), str(f)) else ERASURE)
            for p, f, e in rows]


def restore(engine: Any, item: Unfinished) -> bool:
    """Make an unfinished ordinary delete's memory findable again."""
    from superlocalmemory.core.delete_refusal import heal_stale

    return item.origin == DELETE and heal_stale(engine, item.profile_id, item.fact_id)


__all__ = ["DELETE", "ERASURE", "Unfinished", "find", "restore"]
