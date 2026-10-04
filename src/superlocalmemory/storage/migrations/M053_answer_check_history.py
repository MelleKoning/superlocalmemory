# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M053 — the Answer Check history, in ``learning.db``.

WHAT IT ADDS
------------
``answer_check_events``: one row per recall the answer check saw — outcome,
enums, counts and timings. There is deliberately no column that could hold
question text, memory text, memory ids, session ids or agent names.

Timings are milliseconds, numbers only: the whole recall (``total_ms``),
finding memories (``retrieval_ms``), and within it waiting for the query's
embedding (``embed_ms``) and ranking (``rerank_ms``); then the check itself
(``judge_ms``). The stage columns tell a slow answer's owner where its time
went.

``answer_check_erasures``: one row per profile whose history was erased, with
the instant. Every insert of history is conditional on there being no erasure
at or after the event's own time, so a batch that was already in flight in the
daemon when another process (``slm gdpr``) erased the profile cannot bring the
rows back. The row is data about the erasure itself, so erasure keeps it.

WHY learning.db
---------------
Learning-plane tables never share memory.db's recall lock domain (see the
comment above M040 in ``_migration_catalogue``).

NO CHECK CONSTRAINTS
--------------------
Values are validated before they are recorded
(``core.answer_check_history.event_from_response``). A CHECK failure would roll
back a whole batch of otherwise good rows.

DEVELOPMENT STORES
------------------
``embed_ms`` and ``rerank_ms`` were added to this migration before 4.1.20 was
released (no released build ever ran M053: v4.1.19 has no M053). A store
created by an earlier 4.1.20 development build has the table without them;
``verify`` reports that and ``repair`` adds the two nullable columns, so the
earlier DDL fingerprint is allowlisted in ``_migration_internals``.

OLDER BUILDS
------------
Additive: no earlier build names these tables, so ``BREAKING_VERSION`` is 0.
``DOWNGRADE_FLOOR`` is 51, not 52: 4.1.18 ignores these tables exactly as 4.1.19
does, and a floor of 52 would refuse the existing "go back to 4.1.18" path that
M052 (floor 51) deliberately keeps open.
"""

from __future__ import annotations

import sqlite3

NAME = "M053_answer_check_history"
DB_TARGET = "learning"

#: Additive: a build that predates this never reads or writes what it adds.
BREAKING_VERSION = 0

#: ``prepare_downgrade`` may lower the recorded schema version to this.
DOWNGRADE_FLOOR = 51

DDL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS answer_check_events (
    event_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    occurred_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    backend TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT '',
    abstained INTEGER NOT NULL DEFAULT 0,
    abstention_reason TEXT,
    answer_confidence REAL,
    threshold REAL,
    reordered INTEGER NOT NULL DEFAULT 0,
    result_count INTEGER NOT NULL DEFAULT 0,
    query_type TEXT NOT NULL DEFAULT '',
    retrieval_ms REAL,
    judge_ms REAL,
    embed_ms REAL,
    rerank_ms REAL,
    total_ms REAL,
    calibration_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (profile_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_answer_check_events_profile_time
    ON answer_check_events (profile_id, occurred_ms DESC);
CREATE TABLE IF NOT EXISTS answer_check_erasures (
    profile_id TEXT PRIMARY KEY,
    erased_at_ms INTEGER NOT NULL
);
COMMIT;
"""

_TABLES = ("answer_check_events", "answer_check_erasures")
_INDEX = "idx_answer_check_events_profile_time"
#: Columns a development build of this migration did not have (nullable REAL).
_STAGE_COLUMNS = ("embed_ms", "rerank_ms")


def _missing_stage_columns(conn: sqlite3.Connection) -> list[str]:
    have = {row[1] for row in conn.execute("PRAGMA table_info(answer_check_events)")}
    return [c for c in _STAGE_COLUMNS if c not in have]


def verify(conn: sqlite3.Connection) -> bool:
    """Both tables, the index and every timing column exist."""
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}
    if not (all(t in names for t in _TABLES) and _INDEX in names):
        return False
    return not _missing_stage_columns(conn)


def repair(conn: sqlite3.Connection) -> None:
    """Re-create whatever is missing. Every statement is IF NOT EXISTS, and
    the stage columns are added (nullable, so existing rows are untouched)."""
    conn.executescript(DDL)
    for column in _missing_stage_columns(conn):
        conn.execute(f"ALTER TABLE answer_check_events ADD COLUMN {column} REAL")  # noqa: S608
    if not verify(conn):
        raise sqlite3.OperationalError("M053 schema did not reach its required end-state")


def blocks_serving(conn: sqlite3.Connection) -> bool:
    """Never: without these tables the history is simply not saved."""
    return False
