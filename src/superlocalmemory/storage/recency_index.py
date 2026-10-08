# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer "which visible memories are newest?" from an index, not from every row.

WHY
---
The time channel's recency fallback asks for the 50 newest visible facts of a
profile from the last 90 days (``TemporalChannel._recency_fallback``). No index
covered ``created_at``, so SQLite read every fact of the profile -- each row
carries its embedding and both Fisher vectors, about 40 KB -- and sorted them.
Measured on a copy of a 22,175-fact store: 5,763 ms cold and 111 ms warm, the
slowest step of the first recalls after a daemon start.

WHAT THIS DOES
--------------
One covering index led by ``profile_id`` and ``created_at``, carrying
``fact_id`` and the visibility columns (``scope``, ``archive_status``,
``quarantined``), so the query is answered by walking the index backwards and
stopping after 50 rows. Same query: 0.1 ms. It changes no answer: SQLite reads
the same values from the index instead of the row, and the ORDER BY (with its
``fact_id`` tie-break) is unchanged.

Created at every engine start (``IF NOT EXISTS``), only when every column
exists, exactly like ``visibility_index``. Building it reads the table once
(about 3 s on that store), on the first start after the upgrade.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "idx_facts_recent"
COLUMNS = ("profile_id", "created_at", "fact_id", "scope", "archive_status", "quarantined")


def ensure(conn: sqlite3.Connection) -> bool:
    """Create the index when the columns exist. True when it exists afterwards."""
    try:
        have = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
        if not set(COLUMNS) <= have:
            return False
        conn.execute(f"CREATE INDEX IF NOT EXISTS {NAME} ON atomic_facts ({', '.join(COLUMNS)})")
        return True
    except sqlite3.Error as exc:  # never fatal: reads stay correct, only slower
        logger.warning("recency index not created: %s", exc)
        return False


__all__ = ["COLUMNS", "NAME", "ensure"]
