# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Store health in five separate answers, so one cannot hide another.

* ``page_integrity`` — is the file itself sound (SQLite ``quick_check``; only
  with ``pages=True``: on a 2 GB store it reads every page);
* ``relational_integrity`` — foreign-key findings and orphan classes, each with
  what the repair does with it (remove / keep), parentless facts, the erased
  words still stored outside the projections;
* ``source_fidelity`` — facts withheld from answers, and facts that no longer
  say what their memory said (``slm db fidelity``);
* ``projection_readiness`` — the work still owed to keyword, vector and date
  search: pending and failed obligations, queues, unreachable vectors;
* ``active_repair`` — a repair running now, and the last one that ran.

Read-only, counts only. Safe while SLM runs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from superlocalmemory.storage import integrity_scan as scan


def _ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _count(conn: sqlite3.Connection, sql: str) -> int | None:
    try:
        return int(conn.execute(sql).fetchone()[0])
    except sqlite3.OperationalError:
        return None


def _pages(conn: sqlite3.Connection, check: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "freelist_pages": _count(conn, "PRAGMA freelist_count"),
        "page_count": _count(conn, "PRAGMA page_count"),
    }
    if check:
        result = [r[0] for r in conn.execute("PRAGMA quick_check(20)")]
        out["quick_check"] = "ok" if result == ["ok"] else result
    else:
        out["quick_check"] = "not run (add --pages; it reads the whole file)"
    return out


def _relational(conn: sqlite3.Connection) -> dict[str, Any]:
    p = scan.plan(conn)
    removable = sum(o["rows"] for o in p["orphans"] if o["action"] == "remove")
    kept = sum(o["rows"] for o in p["orphans"] if o["action"] == "keep")
    return {
        "foreign_key_findings": p["foreign_key_findings"],
        "foreign_key_total": sum((p["foreign_key_findings"] or {}).values()),
        "orphans": {o["table"]: {"rows": o["rows"], "action": o["action"]} for o in p["orphans"]
                    if o["rows"]},
        "removable_rows": removable,
        "kept_rows": kept,
        "parentless_facts": p["parentless_facts"],
        "erased_text_leftovers": p["erased_text"],
        "keyword_index": p["keyword_index"],
        "_plan": p,
    }


def _fidelity(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        from superlocalmemory.cli.fidelity_cmd import _to_review, _withheld
    except Exception:  # an install without the fidelity check
        return {"available": False}
    withheld, _ = _withheld(conn, "", 0)
    review, _ = _to_review(conn, "", 0)
    return {"withheld_from_answers": withheld, "to_review": review,
            "see": "slm db fidelity"}


def _readiness(conn: sqlite3.Connection, plan: dict) -> dict[str, Any]:
    states: dict[str, int] = {}
    try:
        for state, n in conn.execute(
                "SELECT state, COUNT(*) FROM projection_obligations GROUP BY state"):
            states[str(state)] = int(n)
    except sqlite3.OperationalError:
        pass
    journal: dict[str, int] = {}
    try:
        for state, n in conn.execute(
                "SELECT state, COUNT(*) FROM ingestion_operations GROUP BY state"):
            journal[str(state)] = int(n)
    except sqlite3.OperationalError:
        pass
    return {
        "obligations": states,
        "failed_obligations": plan["failed_obligations"],
        "projection_queue": _count(conn, "SELECT COUNT(*) FROM projection_outbox"),
        "ingestion": journal,
        "facts_without_embedding": _count(
            conn, "SELECT COUNT(*) FROM atomic_facts WHERE embedding IS NULL"),
        "unreachable_vectors": plan["unreachable_vectors"],
    }


def _repair(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        rows = conn.execute("SELECT run_id, status, started_at, finished_at, summary_json FROM "
                            "integrity_repair_runs ORDER BY started_at DESC LIMIT 5").fetchall()
    except sqlite3.OperationalError:
        return {"running": None, "last": None}
    runs = [{"run_id": r[0], "status": r[1], "started_at": r[2], "finished_at": r[3],
             "summary": json.loads(r[4] or "{}")} for r in rows]
    running = next((r for r in runs if r["status"] == "running"), None)
    return {"running": running, "last": runs[0] if runs else None}


def health(db_path: str | Path, *, pages: bool = False) -> dict[str, Any]:
    conn = _ro(Path(db_path))
    try:
        relational = _relational(conn)
        plan = relational.pop("_plan")
        return {
            "page_integrity": _pages(conn, pages),
            "relational_integrity": relational,
            "source_fidelity": _fidelity(conn),
            "projection_readiness": _readiness(conn, plan),
            "active_repair": _repair(conn),
        }
    finally:
        conn.close()


__all__ = ["health"]
