# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The persisted re-index job: one row per switch or rollback.

A job row is what makes a re-index survive a restart: the runner resumes from
``cursor`` (an ``atomic_facts`` rowid). Configs are stored WITHOUT keys
(storage/embedding_spaces.public_config); a hosted model's key lives in a
0600 side file (core/embedding_reindex_secrets.py), never here.
"""

from __future__ import annotations

import json
import time
from typing import Any

from superlocalmemory.storage.embedding_spaces import (
    ACTIVE_STATES,
    JOBS,
    ensure_control_tables,
    table_exists,
)


class JobConflict(RuntimeError):
    """Another job is still active; carries that job so the caller can name it."""

    def __init__(self, job: dict) -> None:
        self.job = job
        super().__init__(
            f"re-index job {job['job_id']} ({job['from_signature']} -> "
            f"{job['to_signature']}) is {job['state']}"
        )


_UPDATABLE = frozenset({
    "state", "cursor", "next_rowid", "total", "done", "copied", "caught_up",
    "attempts", "error", "stats", "started_at", "finished_at", "activated_at",
})


def active_job(conn: Any) -> dict | None:
    if not table_exists(conn, JOBS):
        return None
    marks = ",".join("?" for _ in ACTIVE_STATES)
    row = conn.execute(
        f"SELECT * FROM {JOBS} WHERE state IN ({marks}) ORDER BY job_id LIMIT 1",
        ACTIVE_STATES,
    ).fetchone()
    return dict(row) if row is not None else None


def latest_job(conn: Any) -> dict | None:
    if not table_exists(conn, JOBS):
        return None
    row = conn.execute(f"SELECT * FROM {JOBS} ORDER BY job_id DESC LIMIT 1").fetchone()
    return dict(row) if row is not None else None


def get_job(conn: Any, job_id: int) -> dict | None:
    if not table_exists(conn, JOBS):
        return None
    row = conn.execute(f"SELECT * FROM {JOBS} WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row is not None else None


def create_job(conn: Any, *, kind: str, from_sig: str, to_sig: str,
               from_cfg: dict, to_cfg: dict, total: int) -> dict:
    """Insert a queued job. Caller holds BEGIN IMMEDIATE, so the check is atomic."""
    ensure_control_tables(conn)
    running = active_job(conn)
    if running is not None:
        raise JobConflict(running)
    now = time.time()
    cur = conn.execute(
        f"INSERT INTO {JOBS} (kind, state, from_signature, to_signature, from_config, "
        "to_config, total, created_at, updated_at) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
        (kind, from_sig, to_sig, json.dumps(from_cfg, sort_keys=True),
         json.dumps(to_cfg, sort_keys=True), int(total), now, now),
    )
    return get_job(conn, int(cur.lastrowid))


def update_job(conn: Any, job_id: int, **values: Any) -> None:
    unknown = set(values) - _UPDATABLE
    if unknown:
        raise ValueError(f"unknown job fields: {sorted(unknown)}")
    values["updated_at"] = time.time()
    names = ", ".join(f"{k} = ?" for k in values)
    conn.execute(f"UPDATE {JOBS} SET {names} WHERE job_id = ?", (*values.values(), job_id))


def public_view(job: dict | None, *, now: float | None = None) -> dict | None:
    """What /health, the CLI and the dashboard show. No configs' secrets exist here."""
    if job is None:
        return None
    now = time.time() if now is None else now
    total = int(job.get("total") or 0)
    done = min(int(job.get("done") or 0), total) if total else int(job.get("done") or 0)
    eta = None
    started = job.get("started_at")
    stats = json.loads(job["stats"]) if job.get("stats") else {}
    this_run = done - int(stats.get("resumed_at_done") or 0)
    if job["state"] in ("running", "catching_up") and started and this_run > 0 and total > done:
        rate = this_run / max(now - float(started), 1e-6)
        eta = round((total - done) / rate, 1) if rate > 0 else None
    stats.pop("_samples", None)
    return {
        "job_id": job["job_id"], "kind": job["kind"], "state": job["state"],
        "from": job["from_signature"], "to": job["to_signature"],
        "done": done, "total": total, "copied": int(job.get("copied") or 0),
        "caught_up": int(job.get("caught_up") or 0), "eta_seconds": eta,
        "error": job.get("error"), "created_at": job["created_at"],
        "started_at": started, "finished_at": job.get("finished_at"),
        "activated_at": job.get("activated_at"), "stats": stats,
    }


__all__ = ["JobConflict", "active_job", "create_job", "get_job", "latest_job",
           "public_view", "update_job"]
