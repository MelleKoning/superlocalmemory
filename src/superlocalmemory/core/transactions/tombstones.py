# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Projection tombstones: the durable record that a fact was erased.

Split out of ``erasure`` so that module stays within the file-size limit.
Everything here is re-exported from ``erasure`` for existing importers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("superlocalmemory.core.transactions.erasure")


# Tombstone write outcomes. WRITTEN/ABSENT are non-failures (ABSENT = the
# table does not exist on an older schema, or there were no fact ids);
# CONFLICT and ERROR must fail the erasure closed on a modern store.
TOMBSTONE_WRITTEN = "written"
TOMBSTONE_ABSENT = "absent"
TOMBSTONE_CONFLICT = "conflict"
TOMBSTONE_ERROR = "error"


def write_tombstones_status(
    db: object,
    profile_id: str,
    fact_ids: tuple[str, ...],
    erasure_id: str,
    created_at: float,
    memory_id: str | None = None,
) -> str:
    """Write projection tombstones and report a precise outcome code.

    Returns one of TOMBSTONE_WRITTEN / TOMBSTONE_ABSENT / TOMBSTONE_CONFLICT /
    TOMBSTONE_ERROR so callers can fail an erasure closed on a real failure
    while still tolerating an older schema without the tombstone table.
    """
    if not fact_ids:
        return TOMBSTONE_ABSENT
    try:
        with db.raw_connection() as conn:
            if not _table_exists(conn, "projection_tombstones"):
                return TOMBSTONE_ABSENT
            for fact_id in fact_ids:
                conn.execute(
                    "INSERT INTO projection_tombstones "
                    "(profile_id, fact_id, erasure_id, memory_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(profile_id, fact_id) DO UPDATE SET "
                    "memory_id = COALESCE(projection_tombstones.memory_id, excluded.memory_id)",
                    (profile_id, fact_id, erasure_id, memory_id, created_at),
                )
                stored = conn.execute(
                    "SELECT memory_id FROM projection_tombstones "
                    "WHERE profile_id = ? AND fact_id = ?",
                    (profile_id, fact_id),
                ).fetchone()
                if (
                    stored is not None
                    and memory_id is not None
                    and stored[0] is not None
                    and stored[0] != memory_id
                ):
                    logger.error(
                        "tombstone provenance conflict for %s: stored=%r != passed=%r; "
                        "failing closed",
                        fact_id[:16], stored[0], memory_id,
                    )
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001
                        pass
                    return TOMBSTONE_CONFLICT
            conn.commit()
        return TOMBSTONE_WRITTEN
    except Exception as exc:  # noqa: BLE001
        logger.warning("erasure tombstone write skipped: %s", _err(exc))
        return TOMBSTONE_ERROR


def write_tombstones(
    db: object,
    profile_id: str,
    fact_ids: tuple[str, ...],
    erasure_id: str,
    created_at: float,
    memory_id: str | None = None,
) -> bool:
    """Backward-compatible boolean wrapper: True only on a clean write."""
    return (
        write_tombstones_status(
            db, profile_id, fact_ids, erasure_id, created_at, memory_id
        )
        == TOMBSTONE_WRITTEN
    )


def is_tombstoned(conn: object, profile_id: str, fact_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM projection_tombstones WHERE profile_id = ? AND fact_id = ?",
        (profile_id, fact_id),
    ).fetchone()
    return row is not None


def tombstone_memory_id(db: object, profile_id: str, fact_id: str) -> str | None:
    try:
        with db.raw_connection() as conn:
            if not _table_exists(conn, "projection_tombstones"):
                return None
            row = conn.execute(
                "SELECT memory_id FROM projection_tombstones "
                "WHERE profile_id = ? AND fact_id = ?",
                (profile_id, fact_id),
            ).fetchone()
            return row[0] if row and row[0] else None
    except Exception:  # noqa: BLE001
        return None


def _table_exists(conn: object, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


__all__ = [
    "TOMBSTONE_ABSENT",
    "TOMBSTONE_CONFLICT",
    "TOMBSTONE_ERROR",
    "TOMBSTONE_WRITTEN",
    "is_tombstoned",
    "tombstone_memory_id",
    "write_tombstones",
    "write_tombstones_status",
]
