# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Putting back the kinds a person confirmed after the copy (L1-09).

Two kinds of edit come out of the export:

  * on a fact the copy already had (``fact_id`` set): the restored row holds the
    copy's OLDER state, possibly an older confirmation of its own.
  * on a fact of a memory added after the copy (``added_memory_id`` set): that
    memory is re-ingested after the restore and its facts get new ids, so the
    edit carries the fact's words and is matched to the re-ingested fact with
    the same words once that memory has finished enriching. Until then it waits
    (it is tried again at the next start), for at most ``MAX_PASSES`` starts.

The newer confirmation wins. An edit is applied unless the row now holds a
confirmation made later than the edit's -- which can only be a person choosing
again after the restore. For a re-ingested fact, only a ``user`` confirmation
counts: its ``caller`` kind was declared by the re-import itself.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

#: Starts a confirmation may wait for its memory to finish enriching.
MAX_PASSES = 10
_WAITING = object()
_UNFINISHED = ("raw", "queryable", "enriching")


def _moment(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _newer(row_at: Any, edit_at: Any) -> bool:
    """True when the row's confirmation is strictly later than the edit's."""
    row, edit = _moment(row_at), _moment(edit_at)
    return row is not None and edit is not None and row > edit


def _words(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _added_target(conn: sqlite3.Connection, edit: dict[str, Any], source_type: str) -> Any:
    """The re-ingested fact this edit belongs to, ``_WAITING``, or None (no match)."""
    op = conn.execute(
        "SELECT state, final_fact_ids_json, queryable_fact_ids_json FROM ingestion_operations "
        "WHERE profile_id=? AND source_type=? AND idempotency_key=?",
        (edit["profile_id"], source_type, f"{source_type}:{edit['added_memory_id']}")).fetchone()
    if op is None:
        return None                       # the memory was not added back (see the report)
    if op[0] in _UNFINISHED:
        return _WAITING
    try:
        ids = [str(i) for i in json.loads(op[1] or "[]") or json.loads(op[2] or "[]")]
    except (TypeError, ValueError):
        return None
    if not ids:
        return None
    marks = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT fact_id, content FROM atomic_facts WHERE profile_id=? "
                        f"AND fact_id IN ({marks})", (edit["profile_id"], *ids)).fetchall()
    want = _words(edit.get("content"))
    same = [r[0] for r in rows if _words(r[1]) == want]
    if len(same) == 1:
        return same[0]
    if not same and len(rows) == 1 and int(edit.get("facts_in_memory") or 0) == 1:
        return rows[0][0]                 # one fact before, one fact now: the same fact
    return None


def _blocked(row: tuple, edit: dict[str, Any], *, added: bool) -> bool:
    _kind, source, at = row
    sources = ("user",) if added else ("user", "caller")
    return source in sources and _newer(at, edit.get("memory_kind_at"))


def reapply_kinds(conn: sqlite3.Connection, edits: list[dict[str, Any]], counts: dict[str, Any],
                  *, source_type: str, history: bool, give_up: bool) -> None:
    """Apply ``edits`` on ``conn`` (the caller holds the write lock and commits)."""
    from superlocalmemory.storage.memory_kinds import COARSE, parse_kind

    now = datetime.now(UTC).isoformat(timespec="seconds")
    for edit in edits:
        kind = parse_kind(edit.get("memory_kind"))
        added = bool(edit.get("added_memory_id"))
        if kind is None or edit.get("memory_kind_source") not in ("user", "caller"):
            counts["kinds_skipped"] += 1
            continue
        target = _added_target(conn, edit, source_type) if added else edit.get("fact_id")
        if target is _WAITING:
            counts["kinds_skipped" if give_up else "kinds_waiting"] += 1
            continue
        row = conn.execute(
            "SELECT memory_kind, memory_kind_source, memory_kind_at FROM atomic_facts "
            "WHERE fact_id=? AND profile_id=?", (target, edit["profile_id"])).fetchone() \
            if target else None
        if row is None or _blocked(row, edit, added=added):
            counts["kinds_skipped"] += 1
            continue
        conn.execute(
            "UPDATE atomic_facts SET memory_kind=?, memory_kind_source=?, "
            "memory_kind_confidence=?, memory_kind_recipe=?, memory_kind_at=?, fact_type=? "
            "WHERE fact_id=? AND profile_id=?",
            (kind.value, edit["memory_kind_source"], edit.get("memory_kind_confidence"),
             edit.get("memory_kind_recipe"), edit.get("memory_kind_at") or now,
             COARSE[kind], target, edit["profile_id"]))
        counts["kinds_reapplied"] += 1
        if history:
            conn.execute(
                "INSERT INTO memory_kind_history (fact_id, profile_id, run_id, origin, "
                "new_kind, new_source, new_fact_type, actor, changed_at) "
                "VALUES (?, ?, NULL, 'restore', ?, ?, ?, 'restore', ?)",
                (target, edit["profile_id"], kind.value, edit["memory_kind_source"],
                 COARSE[kind], now))


__all__ = ["MAX_PASSES", "reapply_kinds"]
