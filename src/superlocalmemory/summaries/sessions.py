# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Which sessions have memories a session summary could be built from (#113).

A session summary needs a session id, and nothing listed them: ``slm summary
session <id>`` told the user to look in ``slm status``, which never showed one,
and the dashboard offered no session summary at all. This is the list.

Read-only, profile-scoped, and limited to memories a person may be shown — the
same visibility rule the summaries themselves apply — so a session holding only
withheld rows is not offered as something to summarise.

Order is newest activity first, then session id, so two calls on an unchanged
store return the same list.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("superlocalmemory.summaries.sessions")

#: Most sessions a caller may ask for at once.
MAX_SESSIONS = 100
DEFAULT_SESSIONS = 20


def list_recent_sessions(
    db_path: str | Path, profile_id: str, limit: int = DEFAULT_SESSIONS,
) -> list[dict[str, Any]]:
    """Recent sessions with at least one visible memory, newest first.

    Each entry: ``session_id``, ``memory_count``, ``first_at``, ``last_at``.
    A store that does not exist yet has no sessions; a read error is logged and
    raised, because "you have no sessions" would be a different, false answer.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_SESSIONS:
        raise ValueError(f"limit must be a whole number from 1 to {MAX_SESSIONS}")
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    from superlocalmemory.storage.database import visible_fact_clause_for_connection

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        visible = visible_fact_clause_for_connection(conn)
        rows = conn.execute(
            f"""
            SELECT session_id, COUNT(*) AS n, MIN(created_at), MAX(created_at)
              FROM atomic_facts
             WHERE profile_id = ?
               AND session_id IS NOT NULL AND session_id != ''
               AND COALESCE(lifecycle, '') != 'archived'{visible}
             GROUP BY session_id
             ORDER BY MAX(created_at) DESC, session_id ASC
             LIMIT ?
            """,  # noqa: S608 - the clause is built from constants only
            (profile_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"session_id": r[0], "memory_count": int(r[1]), "first_at": r[2], "last_at": r[3]}
        for r in rows
    ]


__all__ = ["DEFAULT_SESSIONS", "MAX_SESSIONS", "list_recent_sessions"]
