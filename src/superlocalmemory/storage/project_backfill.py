# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Give session-end summaries the project their own text names (GitHub #150).

Before 4.1.21 the Stop hook saved "[superlocalmemory] session ended ..." with
no project, so recall's project preference, the project filter and the
project summary could not see those memories even though the project was
written in their first word. This pass copies that name into the memory's
saved project (``memories.metadata_json -> project``), the same field
``remember(project=...)`` writes, and marks where it came from
(``_slm_project_source = "session_end_prefix"``).

Why a maintenance pass and not a migration: a migration on this table would
make the upgrade copy the whole memory.db as a restore point first, which
4.1.20 deliberately stopped doing. This needs no new column - only a key in
JSON every row already has - so it runs in the background, in batches:

* idempotent - a row that already has a project is never touched, so a row a
  person or agent tagged keeps their value, and a second run changes nothing;
* bounded by data, not time - at most ``max_batches`` x ``batch_size`` rows a
  pass, each batch in its own short write transaction, so a memory being
  saved waits for one batch, never the sweep;
* profile-preserving - each row is updated in place, inside its own profile.

Rows whose text names no project are left alone (the issue's "no derivable
project" case): they stay findable through every unfiltered recall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, ContextManager

from superlocalmemory.core.project_identity import session_end_project

logger = logging.getLogger(__name__)

SOURCE_KEY = "_slm_project_source"
SOURCE_SESSION_END = "session_end_prefix"

#: Rows read per batch, and batches per pass.
BATCH_SIZE = 200
MAX_BATCHES = 25

# Cheap pre-filter; session_end_project() decides. No project yet, valid JSON.
_SELECT = (
    "SELECT rowid AS rid, memory_id, content FROM memories "
    "WHERE rowid > ? AND content LIKE '[%] session ended %' "
    "AND json_valid(metadata_json) "
    "AND trim(COALESCE(json_extract(metadata_json, '$.project'), '')) = '' "
    "ORDER BY rowid LIMIT ?"
)
_UPDATE = (
    "UPDATE memories SET metadata_json = json_set(metadata_json, '$.project', ?, "
    f"'$.{SOURCE_KEY}', ?) "
    "WHERE memory_id = ? "
    "AND trim(COALESCE(json_extract(metadata_json, '$.project'), '')) = ''"
)


@dataclass(frozen=True, slots=True)
class BackfillReport:
    scanned: int
    tagged: int
    finished: bool


def _one_batch(conn: Any, after: int, batch_size: int) -> tuple[int, int, int]:
    """(rows read, rows tagged, last rowid read) for one batch."""
    rows = conn.execute(_SELECT, (after, batch_size)).fetchall()
    tagged = 0
    last = after
    for row in rows:
        rid, memory_id, content = row[0], row[1], row[2]
        last = int(rid)
        name = session_end_project(content)
        if name is None:
            continue
        cur = conn.execute(_UPDATE, (name, SOURCE_SESSION_END, memory_id))
        tagged += int(cur.rowcount or 0)
    return len(rows), tagged, last


def backfill_session_end_projects(
    open_connection: Callable[[], ContextManager[Any]], *,
    batch_size: int = BATCH_SIZE, max_batches: int = MAX_BATCHES,
) -> BackfillReport:
    """Tag up to ``max_batches`` batches of untagged session-end summaries.

    ``open_connection`` is ``DatabaseManager.raw_connection``: entering it
    takes the single-writer lock, and leaving it commits, so each batch holds
    the lock only for itself. ``finished`` is True once a batch came back
    short, i.e. nothing is left to look at. Raises what the store raises; the
    scheduler records it as a failed step.
    """
    after, scanned, tagged = 0, 0, 0
    for _ in range(max(1, int(max_batches))):
        with open_connection() as conn:
            read, done, after = _one_batch(conn, after, int(batch_size))
        scanned += read
        tagged += done
        if read < batch_size:
            return BackfillReport(scanned, tagged, True)
    return BackfillReport(scanned, tagged, False)


__all__ = ["BATCH_SIZE", "BackfillReport", "MAX_BATCHES", "SOURCE_KEY",
           "SOURCE_SESSION_END", "backfill_session_end_projects"]
