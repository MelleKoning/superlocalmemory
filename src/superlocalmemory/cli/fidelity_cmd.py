# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm db fidelity`` — facts that do not say what their memory said.

    slm db fidelity [--profile P] [--limit N] [--json]
    slm db fidelity --withhold FACT_ID     (with SLM stopped)
    slm db fidelity --release FACT_ID      (with SLM stopped)

The listing is read-only and safe while the daemon runs. It shows two groups:

* **withheld** — derived facts the write path kept out of answers because they
  changed the source (a number read as a date, an invented measurement, a lost
  "never");
* **to review** — older facts, written before that check existed, that fail it
  today. Nothing about them is changed automatically: a model must not rewrite
  what you stored. ``--withhold FACT_ID`` takes one out of answers (undo with
  ``--release``); or correct it with ``slm update FACT_ID "..."`` (a
  reviewable, undoable correction) and finish with ``slm review-correction``.

``--release`` puts a withheld fact back into answers when you judge it correct.
It is recorded, so the same fact is not withheld again.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from superlocalmemory.core.source_fidelity_guard import REASON_PREFIX, RELEASED_PREFIX
from superlocalmemory.encoding.source_fidelity import check_fact_against_source


def register_db_fidelity_parser(db_sub: Any) -> None:
    """Attach ``slm db fidelity``. Called from cli/main.py."""
    p = db_sub.add_parser(
        "fidelity",
        help="List facts that changed what their memory said (numbers, dates, 'never')",
    )
    p.add_argument("--profile", default="", help="Workspace to check. Default: all")
    p.add_argument("--limit", type=int, default=50, help="Rows to show per group")
    p.add_argument("--release", default="", metavar="FACT_ID",
                   help="Put a withheld fact back into answers (SLM must be stopped)")
    p.add_argument("--withhold", default="", metavar="FACT_ID",
                   help="Take a listed fact out of answers, undoably (SLM must be stopped)")
    p.add_argument("--json", action="store_true", help="machine-readable output")


def _connect_read_only(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _withheld(conn: sqlite3.Connection, profile: str, limit: int) -> tuple[int, list[dict]]:
    if "derivation_lineage" not in {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }:
        return 0, []
    sql = (
        "SELECT DISTINCT l.object_id AS fact_id, l.profile_id, l.unresolved_reason "
        "FROM derivation_lineage l JOIN atomic_facts f ON f.fact_id = l.object_id "
        "WHERE l.object_type = 'fact' AND l.unresolved_reason LIKE ? "
        "AND COALESCE(f.quarantined, 0) = 1"
    )
    params: list[Any] = [REASON_PREFIX + "%"]
    if profile:
        sql += " AND l.profile_id = ?"
        params.append(profile)
    rows = [dict(r) for r in conn.execute(sql + " ORDER BY l.created_at", params)]
    return len(rows), [
        {"fact_id": r["fact_id"], "profile_id": r["profile_id"],
         "reasons": r["unresolved_reason"][len(REASON_PREFIX):].split("+")}
        for r in rows[:limit]
    ]


def _to_review(conn: sqlite3.Connection, profile: str, limit: int) -> tuple[int, list[dict]]:
    """Live facts that fail the check against their parent memory today."""
    live = "COALESCE(f.quarantined, 0) = 0" if "quarantined" in _columns(conn, "atomic_facts") \
        else "1 = 1"
    sql = (
        "SELECT f.fact_id, f.profile_id, f.content, m.content AS source "
        "FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id "
        f"WHERE {live}"
    )
    params: list[Any] = []
    if profile:
        sql += " AND f.profile_id = ?"
        params.append(profile)
    found: list[dict] = []
    total = 0
    for row in conn.execute(sql + " ORDER BY f.created_at", params):
        fact, source = row["content"] or "", row["source"] or ""
        if not fact or not source or fact.strip() in source:
            continue
        report = check_fact_against_source(fact, source)
        if report.ok:
            continue
        total += 1
        if len(found) < limit:
            found.append({"fact_id": row["fact_id"], "profile_id": row["profile_id"],
                          "reasons": list(report.reasons)})
    return total, found


_STOP_FIRST = ("SLM is running. Stop it first (`slm serve stop`), then run "
               "`slm db fidelity` again.")


def _daemon_running() -> bool:
    from superlocalmemory.cli.daemon import owned_daemon_process_alive

    return bool(owned_daemon_process_alive())


def _withhold(db_path: str, fact_id: str) -> tuple[bool, str]:
    """Withhold one fact that fails the check today; recorded so --release undoes it."""
    if _daemon_running():
        return False, _STOP_FIRST
    from superlocalmemory.core.derivation_lineage import _record

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        row = conn.execute(
            "SELECT f.profile_id, f.content, m.content AS source, "
            "COALESCE(f.quarantined, 0) AS quarantined FROM atomic_facts f "
            "JOIN memories m ON m.memory_id = f.memory_id WHERE f.fact_id = ?",
            (fact_id,),
        ).fetchone()
        if row is None:
            return False, f"{fact_id} is not a fact with a source memory in this store."
        if row["quarantined"]:
            return False, f"{fact_id} is already withheld."
        report = check_fact_against_source(row["content"] or "", row["source"] or "")
        if report.ok:
            return False, f"{fact_id} says what its memory said; nothing to withhold."
        with conn:
            conn.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (fact_id,))
            _record(conn, profile_id=row["profile_id"], object_type="fact", object_id=fact_id,
                    operation_id="slm-db-fidelity", derivation_version="review",
                    source_status="unresolved",
                    unresolved_reason=REASON_PREFIX + "+".join(report.reasons))
        return True, f"{fact_id} is out of answers. Undo: slm db fidelity --release {fact_id}"
    finally:
        conn.close()


def _release(db_path: str, fact_id: str) -> tuple[bool, str]:
    if _daemon_running():
        return False, _STOP_FIRST
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        rows = conn.execute(
            "SELECT lineage_id, unresolved_reason FROM derivation_lineage "
            "WHERE object_type='fact' AND object_id=? AND unresolved_reason LIKE ?",
            (fact_id, REASON_PREFIX + "%"),
        ).fetchall()
        if not rows:
            return False, f"{fact_id} was not withheld by the source check; nothing to release."
        with conn:
            conn.execute("UPDATE atomic_facts SET quarantined = 0 WHERE fact_id = ?", (fact_id,))
            for lineage_id, reason in rows:
                conn.execute(
                    "UPDATE derivation_lineage SET unresolved_reason = ? WHERE lineage_id = ?",
                    (RELEASED_PREFIX + reason[len(REASON_PREFIX):], lineage_id),
                )
        return True, f"{fact_id} is back in answers. The release is recorded."
    finally:
        conn.close()


def _print_group(title: str, total: int, rows: list[dict], hint: str) -> None:
    print(f"{title}: {total}")
    for row in rows:
        print(f"  {row['fact_id']}  [{row['profile_id']}]  {', '.join(row['reasons'])}")
    if total > len(rows):
        print(f"  ... {total - len(rows)} more (use --limit)")
    if total:
        print(f"  {hint}")


def cmd_db_fidelity(args: Namespace) -> int:
    """Entry point for ``slm db fidelity``. Returns the process exit code."""
    from superlocalmemory.core.config import SLMConfig

    db_path = str(SLMConfig.load().db_path)
    as_json = bool(getattr(args, "json", False))
    for flag, action in (("release", _release), ("withhold", _withhold)):
        fact_id = str(getattr(args, flag, "") or "").strip()
        if fact_id:
            ok, message = action(db_path, fact_id)
            if as_json:
                print(json.dumps({"ok": ok, "fact_id": fact_id, "message": message}))
            else:
                print(f"[slm] {message}")
            return 0 if ok else 1

    profile = str(getattr(args, "profile", "") or "").strip()
    limit = max(1, int(getattr(args, "limit", 50) or 50))
    if not Path(db_path).exists():
        print(json.dumps({"ok": True, "withheld": {"total": 0, "facts": []},
                          "to_review": {"total": 0, "facts": []}}) if as_json
              else "No memories yet; nothing to check.")
        return 0
    try:
        conn = _connect_read_only(db_path)
    except sqlite3.Error as exc:
        print(f"[slm] could not open the memory store read-only: {exc}", file=sys.stderr)
        return 1
    try:
        withheld_total, withheld = _withheld(conn, profile, limit)
        review_total, review = _to_review(conn, profile, limit)
    finally:
        conn.close()
    if as_json:
        print(json.dumps({
            "ok": True,
            "withheld": {"total": withheld_total, "facts": withheld},
            "to_review": {"total": review_total, "facts": review},
        }))
        return 0
    _print_group("Withheld from answers", withheld_total, withheld,
                 "Correct ones: slm db fidelity --release FACT_ID (with SLM stopped)")
    _print_group("Older facts to review", review_total, review,
                 "Review: slm db fidelity --withhold FACT_ID (undoable), or "
                 'slm update FACT_ID "the right text" then slm review-correction')
    if not withheld_total and not review_total:
        print("Every fact says what its memory said.")
    return 0


__all__ = ["RELEASED_PREFIX", "cmd_db_fidelity", "register_db_fidelity_parser"]
