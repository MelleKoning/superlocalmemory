# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Batch and single-row write operations for memory kinds.

Split out of ``memory_kind_store.py`` purely to keep each file under the
project's per-file size guidance (many small files over one large one) — the
public surface is still exactly ``MemoryKindStore``; these are its write-path
internals, called only from there. Nothing here is part of the frozen LLD
§1 contract and nothing outside ``memory_kinds.py`` spells a kind value as a
literal: every kind in or out is a ``MemoryKind`` member or its ``.value``.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Sequence

from superlocalmemory.storage.memory_kinds import (
    COARSE,
    KindSource,
    MemoryKind,
    is_confirmed,
    kind_fields,
    parse_kind,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.memory_kind_store import BatchResult, KindChange


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return dict(row)


_HISTORY_INSERT_SQL = (
    "INSERT INTO memory_kind_history (fact_id, profile_id, run_id, origin, old_kind, "
    "old_source, old_confidence, old_fact_type, new_kind, new_source, new_confidence, "
    "new_fact_type, actor, changed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

#: A run's own change also keeps the recipe and time it replaced, so undoing
#: a refresh run restores those exactly too, not only the kind.
_BACKFILL_HISTORY_INSERT_SQL = (
    "INSERT INTO memory_kind_history (fact_id, profile_id, run_id, origin, old_kind, "
    "old_source, old_confidence, old_fact_type, old_recipe, old_at, new_kind, new_source, "
    "new_confidence, new_fact_type, actor, changed_at) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


def _autocheckpoint(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("PRAGMA wal_autocheckpoint").fetchone()
        return int(row[0]) if row is not None else 0
    except (sqlite3.Error, TypeError, ValueError):
        return 0


@contextmanager
def _batch_connection(db: "DatabaseManager") -> Iterator[sqlite3.Connection]:
    """A write connection whose automatic checkpoint waits until after the batch.

    SQLite runs its automatic checkpoint inside COMMIT — after the write lock
    is released, but while this process still holds the manager's lock (the
    ``raw_connection`` block). Measured on a copy of a real store, it was most
    of that time (50 rows: 13.7 ms with it, 1.1 ms without). So a batch
    pauses it on its own connection and ``_checkpoint_after`` runs the same
    passive checkpoint once both locks are free. Other connections keep
    checkpointing exactly as before.
    """
    with db.raw_connection() as conn:
        previous = _autocheckpoint(conn)
        if previous:
            conn.execute("PRAGMA wal_autocheckpoint=0")
        try:
            yield conn
        finally:
            if previous:
                try:
                    conn.execute(f"PRAGMA wal_autocheckpoint={int(previous)}")
                except sqlite3.Error:
                    pass


def _checkpoint_after(db: "DatabaseManager") -> None:
    try:
        db.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except sqlite3.Error:
        pass


def _held_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def apply_batch(
    db: "DatabaseManager", run_id: str, profile_id: str, changes: Sequence["KindChange"],
    *, new_cursor: int, actor: str, examined: int | None = None,
) -> "BatchResult":
    """Apply a batch of kind changes and advance the run cursor, atomically.

    Each UPDATE is guarded by the kind values it was read with: a fact a
    concurrent writer touched since selection is skipped, not clobbered. If
    the run is no longer `running` (paused/cancelled meanwhile) the whole
    batch is rolled back and reported inactive, never raised.

    ``examined`` (facts the batch looked at, changed or not) lets the run's
    ``processed``/``skipped`` counters move in this same transaction; without
    it ``processed`` grows by ``len(changes)`` and ``skipped`` is left alone.
    """
    from superlocalmemory.storage.memory_kind_store import BatchResult

    if not changes:
        return BatchResult(applied=0, skipped=0, cursor=new_cursor, run_still_active=True)
    if not db.has_memory_kind_columns():
        return BatchResult(
            applied=0, skipped=len(changes), cursor=new_cursor, run_still_active=False,
        )

    now = _now_iso()
    applied = 0
    skipped = 0
    with _batch_connection(db) as conn:
        started = time.perf_counter()
        conn.execute("BEGIN IMMEDIATE")
        try:
            for change in changes:
                cols = change.new.as_columns(now)
                confirmed_new = is_confirmed(change.new.source.value)
                new_fact_type = COARSE[change.new.kind] if confirmed_new else change.fact_type
                prior = conn.execute(
                    "SELECT memory_kind_recipe, memory_kind_at FROM atomic_facts "
                    "WHERE rowid = ? AND profile_id = ?", (change.rowid, profile_id),
                ).fetchone()
                old_recipe, old_at = (prior[0], prior[1]) if prior is not None else (None, None)
                cur = conn.execute(
                    "UPDATE atomic_facts SET memory_kind = ?, memory_kind_source = ?, "
                    "memory_kind_confidence = ?, memory_kind_recipe = ?, memory_kind_at = ?, "
                    "fact_type = ? WHERE rowid = ? AND profile_id = ? "
                    "AND memory_kind IS ? AND memory_kind_source IS ?",
                    (
                        cols["memory_kind"], cols["memory_kind_source"],
                        cols["memory_kind_confidence"], cols["memory_kind_recipe"],
                        cols["memory_kind_at"], new_fact_type,
                        change.rowid, profile_id, change.old_kind, change.old_source,
                    ),
                )
                if cur.rowcount == 1:
                    applied += 1
                    conn.execute(
                        _BACKFILL_HISTORY_INSERT_SQL,
                        (
                            change.fact_id, profile_id, run_id, "backfill",
                            change.old_kind, change.old_source, change.old_confidence,
                            change.fact_type, old_recipe, old_at,
                            cols["memory_kind"], cols["memory_kind_source"],
                            cols["memory_kind_confidence"], new_fact_type, actor, now,
                        ),
                    )
                else:
                    skipped += 1

            looked_at = len(changes) if examined is None else max(examined, len(changes))
            not_changed = 0 if examined is None else looked_at - applied
            run_cur = conn.execute(
                "UPDATE memory_kind_runs SET cursor_rowid = ?, processed = processed + ?, "
                "changed = changed + ?, skipped = skipped + ?, updated_at = ? "
                "WHERE run_id = ? AND status = 'running'",
                (new_cursor, looked_at, applied, not_changed, now, run_id),
            )
            if run_cur.rowcount != 1:
                conn.execute("ROLLBACK")
                result = BatchResult(
                    applied=0, skipped=len(changes), cursor=new_cursor, run_still_active=False,
                    write_ms=_held_ms(started),
                )
            else:
                conn.commit()
                result = BatchResult(applied=applied, skipped=skipped, cursor=new_cursor,
                                     run_still_active=True, write_ms=_held_ms(started))
        except Exception:
            conn.rollback()
            raise
    _checkpoint_after(db)
    return result


def revert_batch(
    db: "DatabaseManager", run_id: str, profile_id: str, *, limit: int = 200,
) -> "BatchResult":
    """Undo up to ``limit`` of a run's backfill changes, newest first.

    Guarded like a forward apply: a fact a user has since retyped by hand no
    longer carries the `new_kind`/`new_source` this run set, so its guarded
    UPDATE matches no row and the user's edit is left untouched.
    """
    from superlocalmemory.storage.memory_kind_store import BatchResult

    if not db.has_memory_kind_columns():
        return BatchResult(applied=0, skipped=0, cursor=0, run_still_active=False)

    now = _now_iso()
    with _batch_connection(db) as conn:
        started = time.perf_counter()
        conn.execute("BEGIN IMMEDIATE")
        try:
            run_row = conn.execute(
                "SELECT revert_cursor FROM memory_kind_runs WHERE run_id = ?", (run_id,),
            ).fetchone()
            if run_row is None:
                conn.execute("ROLLBACK")
                return BatchResult(applied=0, skipped=0, cursor=0, run_still_active=False)
            revert_cursor = run_row["revert_cursor"]
            if revert_cursor is None:
                revert_cursor = 1 << 62  # unset: walk from the newest history row

            rows = conn.execute(
                "SELECT history_id, fact_id, old_kind, old_source, old_confidence, "
                "old_recipe, old_at, new_kind, new_source FROM memory_kind_history "
                "WHERE run_id = ? AND profile_id = ? AND origin = 'backfill' "
                "AND history_id < ? ORDER BY history_id DESC LIMIT ?",
                (run_id, profile_id, revert_cursor, limit),
            ).fetchall()

            applied = 0
            skipped = 0
            last_history_id = revert_cursor
            for row in rows:
                d = _row_dict(row)
                last_history_id = d["history_id"]
                # Undo restores all five kind columns exactly as they were:
                # an untyped row gets NULLs back, a refreshed one its earlier
                # recipe and time.
                cur = conn.execute(
                    "UPDATE atomic_facts SET memory_kind = ?, memory_kind_source = ?, "
                    "memory_kind_confidence = ?, memory_kind_recipe = ?, memory_kind_at = ? "
                    "WHERE fact_id = ? AND profile_id = ? "
                    "AND memory_kind IS ? AND memory_kind_source IS ?",
                    (
                        d["old_kind"], d["old_source"], d["old_confidence"],
                        d["old_recipe"], d["old_at"],
                        d["fact_id"], profile_id, d["new_kind"], d["new_source"],
                    ),
                )
                if cur.rowcount == 1:
                    applied += 1
                    conn.execute(
                        _HISTORY_INSERT_SQL,
                        (
                            d["fact_id"], profile_id, run_id, "revert",
                            d["new_kind"], d["new_source"], None, None,
                            d["old_kind"], d["old_source"], d["old_confidence"], None,
                            "system", now,
                        ),
                    )
                else:
                    skipped += 1

            finished = len(rows) < limit
            status_sql = (
                "UPDATE memory_kind_runs SET revert_cursor = ?, status = 'reverted', "
                "finished_at = ?, updated_at = ? WHERE run_id = ?"
                if finished else
                "UPDATE memory_kind_runs SET revert_cursor = ?, updated_at = ? WHERE run_id = ?"
            )
            status_params = (
                (last_history_id, now, now, run_id) if finished
                else (last_history_id, now, run_id)
            )
            conn.execute(status_sql, status_params)
            conn.commit()
            held = _held_ms(started)
        except Exception:
            conn.rollback()
            raise
    _checkpoint_after(db)
    return BatchResult(
        applied=applied, skipped=skipped, cursor=last_history_id, run_still_active=not finished,
        write_ms=held,
    )


def reconcile_confirmed(db: "DatabaseManager", profile_id: str, *, limit: int = 500) -> int:
    """Repair ``fact_type`` on confirmed rows a 4.1.18 downgrade window changed (I2)."""
    if not db.has_memory_kind_columns():
        return 0
    now = _now_iso()
    fixed = 0
    with db.raw_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT rowid, fact_id, memory_kind, memory_kind_source, "
                "memory_kind_confidence, fact_type FROM atomic_facts "
                "WHERE profile_id = ? AND memory_kind_source IN ('user', 'caller') "
                "AND memory_kind IS NOT NULL LIMIT ?",
                (profile_id, limit),
            ).fetchall()
            for row in rows:
                d = _row_dict(row)
                parsed = parse_kind(d["memory_kind"])
                if parsed is None:
                    continue
                expected = COARSE[parsed]
                if d["fact_type"] == expected:
                    continue
                cur = conn.execute(
                    "UPDATE atomic_facts SET fact_type = ? WHERE rowid = ? "
                    "AND profile_id = ? AND memory_kind IS ? AND memory_kind_source IS ?",
                    (expected, d["rowid"], profile_id, d["memory_kind"], d["memory_kind_source"]),
                )
                if cur.rowcount == 1:
                    fixed += 1
                    conn.execute(
                        _HISTORY_INSERT_SQL,
                        (
                            d["fact_id"], profile_id, None, "reconcile",
                            d["memory_kind"], d["memory_kind_source"],
                            d["memory_kind_confidence"], d["fact_type"],
                            d["memory_kind"], d["memory_kind_source"],
                            d["memory_kind_confidence"], expected, "system", now,
                        ),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return fixed


def set_kinds(
    db: "DatabaseManager", conn: sqlite3.Connection, profile_id: str,
    items: Sequence[tuple[str, MemoryKind]], *, actor: str,
) -> list[dict[str, Any]]:
    """Confirm a kind for one or more facts, inside the caller's own transaction.

    Neither begins nor commits — the coordinator owns that lifecycle.
    Owner-profile only: a fact owned by another profile is reported
    not-found rather than retyped (R24).
    """
    if not db.has_memory_kind_columns():
        return [
            {"fact_id": fact_id, "ok": False, "error": "memory kinds are not available yet"}
            for fact_id, _kind in items
        ]
    now = _now_iso()
    results: list[dict[str, Any]] = []
    for fact_id, kind in items:
        parsed = kind if isinstance(kind, MemoryKind) else parse_kind(kind)
        if parsed is None:
            results.append({"fact_id": fact_id, "ok": False, "error": "invalid kind"})
            continue
        existing = conn.execute(
            "SELECT memory_kind, memory_kind_source, memory_kind_confidence, fact_type "
            "FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        ).fetchone()
        if existing is None:
            results.append({"fact_id": fact_id, "ok": False, "error": "not found"})
            continue
        prior = _row_dict(existing)
        new_fact_type = COARSE[parsed]
        cur = conn.execute(
            "UPDATE atomic_facts SET memory_kind = ?, memory_kind_source = ?, "
            "memory_kind_confidence = NULL, memory_kind_recipe = NULL, memory_kind_at = ?, "
            "fact_type = ? WHERE fact_id = ? AND profile_id = ?",
            (parsed.value, KindSource.USER.value, now, new_fact_type, fact_id, profile_id),
        )
        if cur.rowcount != 1:
            results.append({"fact_id": fact_id, "ok": False, "error": "not found"})
            continue
        conn.execute(
            _HISTORY_INSERT_SQL,
            (
                fact_id, profile_id, None, "user_edit",
                prior.get("memory_kind"), prior.get("memory_kind_source"),
                prior.get("memory_kind_confidence"), prior.get("fact_type"),
                parsed.value, KindSource.USER.value, None, new_fact_type, actor, now,
            ),
        )
        results.append({
            "fact_id": fact_id, "ok": True,
            **kind_fields({
                "memory_kind": parsed.value, "memory_kind_source": KindSource.USER.value,
                "memory_kind_confidence": None, "fact_type": new_fact_type,
            }),
        })
    return results
