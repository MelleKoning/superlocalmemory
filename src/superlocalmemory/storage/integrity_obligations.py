# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Failed projection obligations, settled by proof and never by retrying all.

A failed obligation is one of four things, decided from the store itself:

* ``proven_erased`` — a fact erase whose subject is tombstoned, gone from the
  canonical table and from that owner's projection: the work is done, only the ledger says
  otherwise. Settled as ``erased``.
* ``obsolete`` — an apply whose subject no longer exists at all: nothing is
  left to project. Settled as ``compensated``.
* ``retryable`` — failed, not cancelled by an administrator, subject still
  there: put back to ``pending`` for the normal redrive.
* ``needs_review`` — cancelled by an administrator while its subject still
  exists, or a residue still present: left exactly as it is and reported.

The ledger row is kept; its ``detail`` keeps the administrator's cancel note
and gains the proof. The old state and detail are kept as an undo copy.
"""

from __future__ import annotations

import json
import sqlite3

PROVEN_ERASED = "proven_erased"
OBSOLETE = "obsolete"
RETRYABLE = "retryable"
NEEDS_REVIEW = "needs_review"

#: Owner -> (table, column) whose row would be the projection residue.
_RESIDUE = {
    "bm25": (("bm25_tokens", "fact_id"),),
    "vector": (("vector_row_map", "fact_id"), ("embedding_metadata", "fact_id")),
    "temporal": (("temporal_events", "fact_id"),),
}


def _exists(conn: sqlite3.Connection, table: str, column: str, value: str) -> bool:
    try:
        return conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1",  # noqa: S608
                            (value,)).fetchone() is not None
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return False
        raise


def _subject_live(conn: sqlite3.Connection, subject: str) -> bool:
    return (_exists(conn, "atomic_facts", "fact_id", subject)
            or _exists(conn, "atomic_facts", "memory_id", subject)
            or _exists(conn, "memories", "memory_id", subject))


def _cancelled(detail: str | None) -> bool:
    try:
        return bool(json.loads(detail or "{}").get("admin_cancel"))
    except (TypeError, ValueError, AttributeError):
        return False


def classify(conn: sqlite3.Connection, row: dict) -> tuple[str, dict]:
    """``(class, proof)`` for one failed obligation row."""
    subject, owner = str(row["subject_id"]), str(row["owner"])
    live = _subject_live(conn, subject)
    residue = [t for t, c in _RESIDUE.get(owner, ()) if _exists(conn, t, c, subject)]
    # Only a fact erasure has a tombstone and a per-fact residue to check; an
    # entity or profile erasure names something else and is left for review.
    tombstoned = _exists(conn, "projection_tombstones", "fact_id", subject)
    proof = {"subject_live": live, "residue": residue, "owner": owner, "kind": row["kind"],
             "tombstoned": tombstoned}
    if row["kind"] == "erase":
        if tombstoned and not live and not residue and owner in _RESIDUE:
            return PROVEN_ERASED, proof
        return NEEDS_REVIEW, proof
    if not live:
        return OBSOLETE, proof
    return (NEEDS_REVIEW if _cancelled(row.get("detail")) else RETRYABLE), proof


def failed_obligations(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    try:
        cur = conn.execute(
            "SELECT obligation_id, operation_id, owner, kind, subject_id, state, detail, attempts "
            "FROM projection_obligations WHERE state = 'failed' ORDER BY obligation_id"
            + (" LIMIT ?" if limit else ""), (limit,) if limit else ())
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return []
        raise
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def census(conn: sqlite3.Connection) -> dict[str, int]:
    out = {PROVEN_ERASED: 0, OBSOLETE: 0, RETRYABLE: 0, NEEDS_REVIEW: 0}
    for row in failed_obligations(conn):
        out[classify(conn, row)[0]] += 1
    return out


def settlement(cls: str, row: dict, proof: dict, run_id: str) -> tuple[str, str, int] | None:
    """``(state, detail, attempts)`` to write, or None to leave the row alone."""
    try:
        detail = json.loads(row.get("detail") or "{}")
        if not isinstance(detail, dict):
            detail = {"previous_detail": detail}
    except (TypeError, ValueError):
        detail = {"previous_detail": row.get("detail")}
    detail = {**detail, "settled_by": f"slm db repair {run_id}", "proof": proof}
    if cls == PROVEN_ERASED:
        return "erased", json.dumps(detail, sort_keys=True), int(row["attempts"])
    if cls == OBSOLETE:
        return "compensated", json.dumps(detail, sort_keys=True), int(row["attempts"])
    if cls == RETRYABLE:
        return "pending", json.dumps(detail, sort_keys=True), 0
    return None


__all__ = ["NEEDS_REVIEW", "OBSOLETE", "PROVEN_ERASED", "RETRYABLE", "census", "classify",
           "failed_obligations", "settlement"]
