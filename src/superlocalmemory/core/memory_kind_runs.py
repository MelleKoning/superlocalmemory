# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Run-row bookkeeping for memory-kind classification runs.

Small SQL helpers the runner (``memory_kind_backfill``) uses for the
``memory_kind_runs`` table: guarded status moves, cursor moves when a batch
had nothing to change, notes, and the read-only queries behind the status
card. Every status move names the states it may leave, so a concurrent
pause/cancel and the runner can never both win.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.core.memory_kind_config import MemoryKindConfig
from superlocalmemory.storage.memory_kind_store import BatchResult

ACTIVE = ("queued", "running", "paused", "reverting")
REVERTIBLE = ("completed", "paused", "cancelled", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def transition(db: Any, run_id: str, to: str, expected: tuple, *,
                started: bool = False, finished: bool = False,
                last_error: str | None = None) -> bool:
    now = _now()
    marks = ", ".join("?" for _ in expected)
    error_sql = "" if last_error is None else ", last_error = ?"
    params: list[Any] = [to, now, int(started), now, int(finished), now]
    if last_error is not None:
        params.append(last_error or None)
    rows = db.execute(
        "UPDATE memory_kind_runs SET status = ?, updated_at = ?, "
        "started_at = CASE WHEN ? = 1 THEN COALESCE(started_at, ?) ELSE started_at END, "
        "finished_at = CASE WHEN ? = 1 THEN ? ELSE finished_at END" + error_sql +
        f" WHERE run_id = ? AND status IN ({marks}) RETURNING run_id",
        (*params, run_id, *expected))
    return bool(rows)

def advance(db: Any, run_id: str, cursor: int, examined: int) -> BatchResult:
    rows = db.execute(
        "UPDATE memory_kind_runs SET cursor_rowid = ?, processed = processed + ?, "
        "skipped = skipped + ?, updated_at = ? WHERE run_id = ? AND status = 'running' "
        "RETURNING run_id", (cursor, examined, examined, _now(), run_id))
    return BatchResult(applied=0, skipped=examined, cursor=cursor,
                       run_still_active=bool(rows))

def note(db: Any, run_id: str, note: str) -> None:
    db.execute("UPDATE memory_kind_runs SET last_error = ?, updated_at = ? "
               "WHERE run_id = ?", (note, _now(), run_id))

def active_run(db: Any, profile_id: str) -> dict | None:
    marks = ", ".join("?" for _ in ACTIVE)
    rows = db.execute(f"SELECT * FROM memory_kind_runs WHERE profile_id = ? AND status "
                      f"IN ({marks}) ORDER BY created_at LIMIT 1", (profile_id, *ACTIVE))
    return dict(rows[0]) if rows else None

def by_source(db: Any, profile_id: str) -> dict[str, int]:
    rows = db.execute(
        "SELECT memory_kind_source AS s, COUNT(*) AS n FROM atomic_facts WHERE "
        "profile_id = ? AND COALESCE(quarantined, 0) = 0 AND memory_kind IS NOT NULL "
        "GROUP BY memory_kind_source", (profile_id,))
    return {str(dict(r)["s"]): int(dict(r)["n"]) for r in rows}

def materializer_due(db: Any) -> bool:
    """New memories waiting for enrichment go first (bounded, see _forward_step)."""
    try:
        from superlocalmemory.core.ingestion_command import (
            _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS as max_attempts,
        )
        now = time.time()
        return bool(db.execute(
            "SELECT 1 FROM ingestion_operations WHERE attempt_count < ? AND "
            "(state = 'queryable' OR (state = 'failed' AND next_retry_at <= ?) OR "
            "(state = 'enriching' AND lease_expires_at <= ?)) LIMIT 1",
            (max_attempts, now, now)))
    except Exception:  # noqa: BLE001 — no queue table, no backlog
        return False

def with_eta(run: dict, cfg: MemoryKindConfig) -> dict:
    out = dict(run)
    remaining = max(0, int(run["total_estimate"]) - int(run["processed"]))
    rate = float(cfg.rate_per_second.get(run["backend"], 8.0))
    try:
        started = datetime.fromisoformat(run["started_at"]) if run["started_at"] else None
    except (TypeError, ValueError):
        started = None
    if started is not None and int(run["processed"]) > 0:
        elapsed = (datetime.now(UTC) - started).total_seconds()
        if elapsed > 0:
            rate = int(run["processed"]) / elapsed
    out["eta_seconds"] = round(remaining / rate, 1) if rate > 0 else None
    return out


def record_error(db: Any, run_id: str, message: str) -> None:
    db.execute("UPDATE memory_kind_runs SET errors = errors + 1, last_error = ?, "
               "updated_at = ? WHERE run_id = ?", (message, _now(), run_id))



def status_view(db: Any, store: Any, cfg: MemoryKindConfig, choice: Any, profile_id: str,
                write_ms: tuple[float, ...], *, not_ready_reason: str) -> dict[str, Any]:
    """The status payload. ``db`` is None when the kind columns are missing."""
    from superlocalmemory.core.memory_kind_backfill_plan import history_rows, recipe_for
    from superlocalmemory.encoding.memory_kind_recipe import calibration_for

    calibration = calibration_for(choice.active, recipe_for(choice.active))
    out: dict[str, Any] = {
        "schema_ready": db is not None, "enabled": cfg.enabled,
        "backend": {"configured": choice.configured, "active": choice.active,
                    "reason": choice.reason, "leaves_device": choice.leaves_device},
        "calibration": {"status": calibration.status,
                        "display_min_confidence": cfg.display_min_confidence},
        "counts": {"kind": {}, "untyped": 0, "legacy": 0, "by_source": {}},
        "active_run": None, "recent_runs": [], "history_rows": 0,
        "batch_write_ms": {"last": write_ms[-1] if write_ms else None,
                           "max": max(write_ms, default=None)},
    }
    if db is None:
        out["reason"] = not_ready_reason
        return out
    counts = store.counts(profile_id, display_min_confidence=cfg.display_min_confidence)
    out["counts"] = {"kind": counts["kind"], "untyped": counts["untyped"],
                     "legacy": counts["legacy"], "by_source": by_source(db, profile_id)}
    active = active_run(db, profile_id)
    out["active_run"] = with_eta(active, cfg) if active else None
    out["recent_runs"] = store.recent_runs(profile_id, limit=5)
    out["history_rows"] = history_rows(db, profile_id)
    return out

__all__ = ["ACTIVE", "REVERTIBLE", "active_run", "advance", "by_source", "materializer_due",
           "note", "record_error", "status_view", "transition", "with_eta"]
