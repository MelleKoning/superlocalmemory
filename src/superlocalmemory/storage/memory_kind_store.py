# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Storage layer for memory-kind backfill runs, reverts and confirmed edits.

Assumes ``DatabaseManager.has_memory_kind_columns()`` may be ``False`` (M052,
WP-1, has not reached this store yet): read methods return an empty/zeroed
result and write methods skip rather than raise (LLD I1, I3).

Owns ``memory_kind_runs`` and ``memory_kind_history`` row access. Never
spells one of the nine kind values as a Python literal — every kind is a
``MemoryKind`` member or its ``.value``. The batch/single-row write bodies
live in the sibling module ``memory_kind_writes.py`` (kept separate only to
stay under this project's per-file size guidance); this file is the public
``MemoryKindStore`` surface every other package codes to.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

from superlocalmemory.storage import memory_kind_writes as _writes
from superlocalmemory.storage.memory_kinds import (
    KindAssignment,
    KindSource,
    MemoryKind,
    is_confirmed,
    kind_fields,
    parse_kind,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from superlocalmemory.storage.database import DatabaseManager


#: Source values a "refresh" backfill pass may re-type. Built from the enum,
#: not spelled twice. USER/CALLER are deliberately excluded (LLD §6.2): a
#: confirmed kind is never silently re-typed by a backfill run.
_REFRESH_ELIGIBLE_SOURCES: tuple[str, ...] = (
    KindSource.RULES.value,
    KindSource.LEGACY.value,
    KindSource.MODEL_LAYA.value,
    KindSource.MODEL_JEV.value,
    KindSource.MODEL_LLM.value,
)


@dataclass(frozen=True, slots=True)
class KindCandidate:
    """One row a backfill pass read and may re-type."""

    rowid: int
    fact_id: str
    profile_id: str
    content: str
    fact_type: str
    memory_kind: str | None
    memory_kind_source: str | None


@dataclass(frozen=True, slots=True)
class KindChange:
    """A proposed kind change for one candidate, carrying the guard values."""

    rowid: int
    fact_id: str
    old_kind: str | None
    old_source: str | None
    old_confidence: float | None
    fact_type: str
    new: KindAssignment


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Outcome of one batch write (apply, revert, or a run-row race)."""

    applied: int
    skipped: int
    cursor: int
    run_still_active: bool


def _row_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return dict(row)


class MemoryKindStore:
    """Read/write access to ``atomic_facts`` kind columns and the two kind tables.

    Callers (WP-4 write path, WP-5 backfill runner and routes) are expected
    to check ``has_memory_kind_columns()`` before doing expensive work, but
    never have to for correctness — every method here is safe to call on a
    store M052 has not reached yet.
    """

    def __init__(self, db: "DatabaseManager") -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Status / counts
    # ------------------------------------------------------------------

    def counts(self, profile_id: str, *, display_min_confidence: float) -> dict[str, Any]:
        """Per-kind confirmed/suggested counts, plus legacy and untyped totals.

        A status-card read (polled every few seconds at most), never the
        recall or remember path — the G4 latency budgets do not apply here.
        """
        kind_counts: dict[str, dict[str, int]] = {
            k.value: {"confirmed": 0, "suggested": 0} for k in MemoryKind
        }
        if not self.db.has_memory_kind_columns():
            return {"schema_ready": False, "kind": kind_counts, "untyped": 0, "legacy": 0}

        rows = self.db.execute(
            "SELECT memory_kind, memory_kind_source, memory_kind_confidence, fact_type "
            "FROM atomic_facts WHERE profile_id = ? AND COALESCE(quarantined, 0) = 0",
            (profile_id,),
        )
        untyped = 0
        legacy = 0
        for row in rows:
            d = _row_dict(row)
            parsed = parse_kind(d.get("memory_kind"))
            source = d.get("memory_kind_source")
            if parsed is not None and is_confirmed(source if isinstance(source, str) else None):
                kind_counts[parsed.value]["confirmed"] += 1
                continue
            if parsed is not None:
                confidence = d.get("memory_kind_confidence")
                try:
                    confidence_f = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    confidence_f = None
                if confidence_f is not None and confidence_f >= display_min_confidence:
                    kind_counts[parsed.value]["suggested"] += 1
                    continue
            fields = kind_fields(d, display_min_confidence=display_min_confidence)
            if fields["memory_kind_state"] == "legacy":
                legacy += 1
            else:
                untyped += 1
        return {"schema_ready": True, "kind": kind_counts, "untyped": untyped, "legacy": legacy}

    def suggestions(
        self, profile_id: str, *, kind: MemoryKind | None, limit: int,
        offset: int, display_min_confidence: float,
    ) -> list[dict[str, Any]]:
        """Facts carrying a model suggestion at or above the display threshold."""
        if not self.db.has_memory_kind_columns():
            return []
        where = [
            "profile_id = ?", "COALESCE(quarantined, 0) = 0",
            "memory_kind IS NOT NULL",
            "memory_kind_source NOT IN ('user', 'caller')",
            "memory_kind_confidence IS NOT NULL",
            "memory_kind_confidence >= ?",
        ]
        params: list[Any] = [profile_id, display_min_confidence]
        if kind is not None:
            where.append("memory_kind = ?")
            params.append(kind.value)
        rows = self.db.execute(
            "SELECT fact_id, content, fact_type, memory_kind, memory_kind_source, "
            "memory_kind_confidence FROM atomic_facts WHERE "
            + " AND ".join(where)
            + " ORDER BY rowid LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            d = _row_dict(row)
            fields = kind_fields(d, display_min_confidence=display_min_confidence)
            out.append({
                "fact_id": d["fact_id"],
                "content_preview": str(d.get("content") or "")[:200],
                **fields,
            })
        return out

    # ------------------------------------------------------------------
    # Backfill batch read
    # ------------------------------------------------------------------

    def select_batch(
        self, profile_id: str, *, after_rowid: int, limit: int,
        mode: Literal["untyped", "refresh"],
    ) -> list[KindCandidate]:
        """Next batch for a backfill run, oldest rowid first, owner profile only.

        ``refresh`` additionally includes rows a non-confirmed source
        classified — never a `user`/`caller` row. Quarantined and tombstoned
        rows are skipped; a store without the kind columns offers nothing.
        """
        if not self.db.has_memory_kind_columns():
            return []
        if mode == "refresh":
            placeholders = ", ".join("?" for _ in _REFRESH_ELIGIBLE_SOURCES)
            mode_clause = f"(memory_kind IS NULL OR memory_kind_source IN ({placeholders}))"
            mode_params: tuple[Any, ...] = _REFRESH_ELIGIBLE_SOURCES
        else:
            mode_clause = "memory_kind IS NULL"
            mode_params = ()

        tombstone_clause = ""
        if self._table_exists("projection_tombstones"):
            tombstone_clause = (
                " AND NOT EXISTS (SELECT 1 FROM projection_tombstones pt "
                "WHERE pt.profile_id = atomic_facts.profile_id "
                "AND pt.fact_id = atomic_facts.fact_id)"
            )

        rows = self.db.execute(
            "SELECT rowid, fact_id, profile_id, content, fact_type, memory_kind, "
            "memory_kind_source FROM atomic_facts "
            "WHERE profile_id = ? AND rowid > ? AND COALESCE(quarantined, 0) = 0 "
            f"AND {mode_clause}{tombstone_clause} ORDER BY rowid LIMIT ?",
            (profile_id, after_rowid, *mode_params, limit),
        )
        return [
            KindCandidate(
                rowid=int(d["rowid"]), fact_id=d["fact_id"], profile_id=d["profile_id"],
                content=d.get("content") or "", fact_type=d.get("fact_type") or "",
                memory_kind=d.get("memory_kind"), memory_kind_source=d.get("memory_kind_source"),
            )
            for d in (_row_dict(r) for r in rows)
        ]

    # ------------------------------------------------------------------
    # Backfill batch / revert / reconcile / confirm — bodies in
    # memory_kind_writes.py; thin delegation keeps this file's public shape
    # exactly the LLD §1 contract.
    # ------------------------------------------------------------------

    def apply_batch(
        self, run_id: str, profile_id: str, changes: Sequence[KindChange],
        *, new_cursor: int, actor: str,
    ) -> BatchResult:
        return _writes.apply_batch(
            self.db, run_id, profile_id, changes, new_cursor=new_cursor, actor=actor,
        )

    def revert_batch(self, run_id: str, profile_id: str, *, limit: int = 200) -> BatchResult:
        return _writes.revert_batch(self.db, run_id, profile_id, limit=limit)

    def reconcile_confirmed(self, profile_id: str, *, limit: int = 500) -> int:
        return _writes.reconcile_confirmed(self.db, profile_id, limit=limit)

    def set_kinds(
        self, conn: sqlite3.Connection, profile_id: str,
        items: Sequence[tuple[str, MemoryKind]], *, actor: str,
    ) -> list[dict[str, Any]]:
        return _writes.set_kinds(self.db, conn, profile_id, items, actor=actor)

    # ------------------------------------------------------------------
    # Run bookkeeping
    # ------------------------------------------------------------------

    def create_run(
        self, profile_id: str, *, backend: str, recipe_id: str, mode: str,
        requested_by: str, total_estimate: int,
    ) -> dict[str, Any]:
        """Queue a new run. One active run per profile is enforced by the
        ``uq_kind_run_active`` partial unique index; a second attempt raises
        ``sqlite3.IntegrityError``, mapped by WP-5's route layer."""
        run_id = uuid.uuid4().hex[:16]
        now = datetime.now(UTC).isoformat()
        self.db.execute(
            "INSERT INTO memory_kind_runs (run_id, profile_id, status, backend, recipe_id, "
            "mode, cursor_rowid, revert_cursor, total_estimate, processed, changed, skipped, "
            "errors, last_error, requested_by, created_at, started_at, finished_at, updated_at) "
            "VALUES (?,?,?,?,?,?,0,NULL,?,0,0,0,0,NULL,?,?,NULL,NULL,?)",
            (run_id, profile_id, "queued", backend, recipe_id, mode, total_estimate,
             requested_by, now, now),
        )
        return self.get_run(run_id) or {}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.db.execute("SELECT * FROM memory_kind_runs WHERE run_id = ?", (run_id,))
        return _row_dict(rows[0]) if rows else None

    def next_runnable(self) -> dict[str, Any] | None:
        """Oldest run still needing work, across every profile."""
        rows = self.db.execute(
            "SELECT * FROM memory_kind_runs WHERE status IN ('queued', 'running', 'reverting') "
            "ORDER BY created_at ASC LIMIT 1"
        )
        return _row_dict(rows[0]) if rows else None

    def set_run_status(
        self, run_id: str, status: str, *, expected: Iterable[str],
        last_error: str | None = None,
    ) -> bool:
        """Transition a run's status, only from one of ``expected``. ``False``
        means another writer already moved it (e.g. a concurrent cancel)."""
        expected_list = list(expected)
        if not expected_list:
            return False
        now = datetime.now(UTC).isoformat()
        placeholders = ", ".join("?" for _ in expected_list)
        if last_error is not None:
            sql = (
                "UPDATE memory_kind_runs SET status = ?, last_error = ?, updated_at = ? "
                f"WHERE run_id = ? AND status IN ({placeholders}) RETURNING run_id"
            )
            params = (status, last_error, now, run_id, *expected_list)
        else:
            sql = (
                "UPDATE memory_kind_runs SET status = ?, updated_at = ? "
                f"WHERE run_id = ? AND status IN ({placeholders}) RETURNING run_id"
            )
            params = (status, now, run_id, *expected_list)
        return bool(self.db.execute(sql, params))

    def recent_runs(self, profile_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM memory_kind_runs WHERE profile_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (profile_id, limit),
        )
        return [_row_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _table_exists(self, name: str) -> bool:
        try:
            rows = self.db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,),
            )
        except sqlite3.Error:
            return False
        return bool(rows)
