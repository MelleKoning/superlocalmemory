# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The user's action wins over a correction only a machine proposed.

Varun's decision (2026-10-06): an explicit delete, replace or edit by the user
or their assistant goes through even when a correction case that SLM itself
proposed, and nobody reviewed, names that fact. That pending case is closed as
"overtaken by a user action": moved, whole and unchanged, to
``correction_cases_overtaken`` with who did what and when (the audit row), and
it can be put back (``restore``) while both of its facts still exist.

Before this, such a case made its facts undeletable forever: the ledger refers
to both facts ``ON DELETE RESTRICT`` and keeps every case. On a real 22k-fact
store 1,692 facts were held only by machine proposals (1,081 cases, all still
``proposed``, none reviewed), and a memory's own facts often proposed to correct
each other.

Who proposed a case is recorded on the case itself: the automatic detectors in
core/store_pipeline.py write ``proposed_by_actor_kind = 'host_attested'`` with
reason ``consolidation_update``, ``consolidation_supersede`` or
``temporal_contradiction``; a person's edit is ``host_authenticated`` with
``direct_content_correction``, and a caller's ``replaces`` is applied at once.
A case counts as machine-proposed only when BOTH say so. A case a person
proposed, or any case already applied (or rejected, or rolled back), protects
exactly as before.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

MACHINE_REASONS = frozenset({"consolidation_update", "consolidation_supersede",
                             "temporal_contradiction"})
MACHINE_ACTOR_KIND = "host_attested"
OVERTAKEN = "overtaken by a user action"

DDL = """
CREATE TABLE IF NOT EXISTS correction_cases_overtaken (
    overtake_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    predecessor_fact_id TEXT NOT NULL,
    successor_fact_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    case_json TEXT NOT NULL,
    events_json TEXT NOT NULL,
    user_action TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    closed_reason TEXT NOT NULL,
    overtaken_at TEXT NOT NULL,
    restored_at TEXT
)
"""


class RestoreRefused(ValueError):
    """The case cannot be put back as it was."""


def _run(target: Any, sql: str, params: tuple = ()) -> list[dict]:
    """Rows as dicts from a sqlite3 connection or a DatabaseManager."""
    result = target.execute(sql, params)
    if hasattr(result, "description"):
        cols = [d[0] for d in result.description] if result.description else []
        return [dict(zip(cols, row)) for row in result.fetchall()]
    return [dict(row) for row in result]


def is_machine_pending(case: dict) -> bool:
    return (str(case.get("status")) == "proposed"
            and str(case.get("reason_code")) in MACHINE_REASONS
            and str(case.get("proposed_by_actor_kind")) == MACHINE_ACTOR_KIND)


def cases_naming(target: Any, fact_ids: Iterable[str], *,
                 predecessor_only: bool = False) -> list[dict]:
    """Every case whose foreign key refers to one of the facts (any profile)."""
    ids = list(dict.fromkeys(str(f) for f in fact_ids))
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    where = f"predecessor_fact_id IN ({ph})" + (
        "" if predecessor_only else f" OR successor_fact_id IN ({ph})")
    params = tuple(ids) if predecessor_only else tuple(ids) * 2
    try:
        return _run(target, f"SELECT * FROM correction_cases WHERE {where} "  # noqa: S608
                    "ORDER BY created_at, case_id", params)
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return []
        raise


def blocking(cases: Iterable[dict]) -> list[dict]:
    """The cases a user action does NOT overtake."""
    return [c for c in cases if not is_machine_pending(c)]


def overtake(target: Any, cases: Iterable[dict], *, user_action: str, actor_id: str,
             operation_id: str) -> list[str]:
    """Close machine-pending cases for a user action. Call inside the action's
    own write transaction so the case closes only if the action happens."""
    closed: list[str] = []
    todo = [c for c in cases if is_machine_pending(c)]
    if not todo:
        return closed
    target.execute(DDL)
    now = datetime.now(timezone.utc).isoformat()
    for case in todo:
        case_id = str(case["case_id"])
        events = _run(target, "SELECT * FROM correction_events WHERE case_id = ? "
                      "ORDER BY system_occurred_at, event_id", (case_id,))
        target.execute(
            "INSERT INTO correction_cases_overtaken (case_id, profile_id, predecessor_fact_id, "
            "successor_fact_id, reason_code, case_json, events_json, user_action, actor_id, "
            "operation_id, closed_reason, overtaken_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (case_id, str(case["profile_id"]), str(case["predecessor_fact_id"]),
             str(case["successor_fact_id"]), str(case["reason_code"]),
             json.dumps(case, sort_keys=True), json.dumps(events, sort_keys=True),
             user_action, actor_id or "unknown", operation_id, OVERTAKEN, now))
        target.execute("DELETE FROM correction_events WHERE case_id = ?", (case_id,))
        target.execute("DELETE FROM correction_cases WHERE case_id = ? AND status = 'proposed'",
                       (case_id,))
        closed.append(case_id)
    logger.info("correction cases overtaken by a user %s: %s", user_action,
                ",".join(c[:12] for c in closed))
    return closed


def _insert(target: Any, table: str, row: dict) -> None:
    cols = list(row)
    target.execute(f"INSERT INTO {table} ({', '.join(cols)}) "  # noqa: S608
                   f"VALUES ({', '.join('?' * len(cols))})", tuple(row[c] for c in cols))


def restore(target: Any, case_id: str) -> dict[str, Any]:
    """Put an overtaken case back, exactly as it was. Idempotent per case."""
    rows = _run(target, "SELECT * FROM correction_cases_overtaken WHERE case_id = ? AND "
                "restored_at IS NULL ORDER BY overtake_id DESC LIMIT 1", (case_id,))
    if not rows:
        raise RestoreRefused(f"no overtaken case {case_id} waiting to be restored")
    row = rows[0]
    case, events = json.loads(row["case_json"]), json.loads(row["events_json"])
    for side in ("predecessor_fact_id", "successor_fact_id"):
        if not _run(target, "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (case[side],)):
            raise RestoreRefused(f"{case[side]} no longer exists (the user removed it), so "
                                 f"case {case_id} cannot come back")
    if _run(target, "SELECT 1 FROM correction_cases WHERE profile_id = ? AND "
            "predecessor_fact_id = ? AND status IN ('proposed', 'applied')",
            (case["profile_id"], case["predecessor_fact_id"])):
        raise RestoreRefused(f"{case['predecessor_fact_id']} has another open correction now")
    _insert(target, "correction_cases", case)
    for event in events:
        _insert(target, "correction_events", event)
    target.execute("UPDATE correction_cases_overtaken SET restored_at = ? WHERE overtake_id = ?",
                   (datetime.now(timezone.utc).isoformat(), row["overtake_id"]))
    return {"restored": case_id, "events": len(events)}


def listing(target: Any, limit: int = 100) -> list[dict]:
    try:
        return _run(target, "SELECT overtake_id, case_id, profile_id, predecessor_fact_id, "
                    "successor_fact_id, reason_code, user_action, actor_id, operation_id, "
                    "closed_reason, overtaken_at, restored_at FROM correction_cases_overtaken "
                    "ORDER BY overtake_id DESC LIMIT ?", (limit,))
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return []
        raise


__all__ = ["MACHINE_REASONS", "OVERTAKEN", "RestoreRefused", "blocking", "cases_naming",
           "is_machine_pending", "listing", "overtake", "restore"]
