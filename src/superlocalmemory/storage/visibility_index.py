# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer "may this profile see this memory?" from an index, not from the row.

WHY
---
Almost every recall stage asks it: the visible count (graph freshness,
Hopfield), which candidate ids may be shown, the correction admission. The
predicate (``_scope_where`` + ``visible_fact_clause``) reads ``profile_id``,
``scope``, ``archive_status`` and ``quarantined``. The last two arrived with
later migrations, so they sit at the END of an ``atomic_facts`` row -- after
the embedding and both Fisher vectors, about 40 KB in. With 4 KB pages, SQLite
has to walk each row's overflow chain to reach them. Measured on a copy of a
22,175-fact store: the visible count took 5,534 ms cold and 111 ms warm, and
the first recalls after a daemon start spent seconds inside such reads.

WHAT THIS DOES
--------------
One covering index over exactly those columns, led by ``profile_id`` and
``fact_id`` so it serves both the per-profile count and the ``fact_id IN (...)``
checks. Same count: 5 ms cold, 3 ms warm. It changes no answer: SQLite reads
the same values from the index instead of the row.

Created at every engine start (``IF NOT EXISTS``), only when every column
exists -- an index on a missing column would stop the engine from starting on
a store a migration has not reached yet. Building it reads the table once
(about 6 s on that store, cold), on the first start after the upgrade.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "idx_facts_visibility"
COLUMNS = ("profile_id", "fact_id", "scope", "archive_status", "quarantined")


def ensure(conn: sqlite3.Connection) -> bool:
    """Create the index when the columns exist. True when it exists afterwards."""
    try:
        have = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
        if not set(COLUMNS) <= have:
            return False
        conn.execute(f"CREATE INDEX IF NOT EXISTS {NAME} ON atomic_facts ({', '.join(COLUMNS)})")
        return True
    except sqlite3.Error as exc:  # never fatal: reads stay correct, only slower
        logger.warning("visibility index not created: %s", exc)
        return False


__all__ = ["COLUMNS", "NAME", "ensure"]
