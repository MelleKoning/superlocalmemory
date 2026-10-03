# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M052 — every memory gets room for its kind, and nothing else changes.

WHAT IT ADDS
------------
Five nullable columns on ``atomic_facts`` — ``memory_kind``,
``memory_kind_source``, ``memory_kind_confidence``, ``memory_kind_recipe``,
``memory_kind_at`` — one partial index over them, and two new tables:
``memory_kind_runs`` (one row per classification run, so a run can pause,
resume and be undone) and ``memory_kind_history`` (one row per change to an
existing memory's kind, which is what makes "undo this run" exact).

NULL means "untyped". Every existing memory starts untyped; its ``fact_type``
is untouched.

WHY ADDITIVE AND NOT A WIDER CHECK
----------------------------------
SQLite's ``ALTER TABLE ... ADD COLUMN`` with no constraint and a NULL default
rewrites only the schema record: no row is read or copied, so it costs the same
on a 50-row store as on an 18,000-row one, and a rowid cannot move. Widening the
``fact_type`` CHECK instead would mean rebuilding the table (the M046 procedure),
which copies every row and is the riskiest operation in this codebase.

There is deliberately no CHECK on any new column. A CHECK on an added column is
tested against every existing row, and — more to the point — a write could then
fail because of a kind, which must never happen (an unknown kind is stored as
untyped, never refused).

ALL OR NOTHING
--------------
Every statement runs through ``conn.execute`` inside one ``BEGIN IMMEDIATE``.
Never ``executescript``: it COMMITs whatever transaction is open, which would
make the rollback below protect nothing. The row count is read before and after
inside the same transaction; if it differs, everything rolls back.

A power cut inside the transaction leaves uncommitted WAL frames that SQLite
discards on the next open, and ``migration_log`` still says ``in_progress``, so
the runner retries; every statement here is idempotent.

EAGER, NOT DEFERRED
-------------------
Registered in ``MIGRATIONS`` so it runs before the engine opens the store and
before journal replay writes a fact. On a fresh install ``atomic_facts`` does not
exist yet: the two tables are created, the ALTERs are skipped, and the engine
creates ``atomic_facts`` with the columns already in place (``storage/schema.py``).

OLDER BUILDS
------------
A 4.1.18 process never names these columns: its explicit-column INSERTs keep
working and its ``SELECT *`` hydration ignores keys it does not know. That is
why ``BREAKING_VERSION`` is 0 and ``DOWNGRADE_FLOOR`` is 51.

Helpers are copied from M046 rather than imported, so this module's recorded
DDL hash never depends on another migration's text.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M052_memory_kinds"
DB_TARGET = "memory"

#: Additive: a build that predates this never reads or writes what it adds.
BREAKING_VERSION = 0

#: ``prepare_downgrade`` may lower the recorded schema version to this.
DOWNGRADE_FLOOR = 51

#: Recorded for the runner's DDL hash. ``apply()`` runs instead of this string,
#: because the ALTERs are conditional on what the live table already has.
DDL = """-- M052: add five nullable memory-kind columns to atomic_facts, one partial index,
-- and the memory_kind_runs / memory_kind_history tables. See apply()."""

_TABLE = "atomic_facts"

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("memory_kind", "TEXT"),
    ("memory_kind_source", "TEXT"),
    ("memory_kind_confidence", "REAL"),
    ("memory_kind_recipe", "TEXT"),
    ("memory_kind_at", "TEXT"),
)

_IDX_FACTS_MEMORY_KIND = (
    "CREATE INDEX IF NOT EXISTS idx_facts_memory_kind "
    "ON atomic_facts (profile_id, memory_kind) WHERE memory_kind IS NOT NULL"
)

_RUNS_TABLE = """CREATE TABLE IF NOT EXISTS memory_kind_runs (
    run_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued','running','paused','completed',
                                           'cancelled','failed','reverting','reverted')),
    backend TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('untyped','refresh')),
    cursor_rowid INTEGER NOT NULL DEFAULT 0,
    revert_cursor INTEGER,
    total_estimate INTEGER NOT NULL DEFAULT 0,
    processed INTEGER NOT NULL DEFAULT 0,
    changed INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    requested_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL)"""

_HISTORY_TABLE = """CREATE TABLE IF NOT EXISTS memory_kind_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    run_id TEXT,
    origin TEXT NOT NULL CHECK (origin IN ('backfill','user_edit','revert',
                                           'reconcile','restore')),
    old_kind TEXT, old_source TEXT, old_confidence REAL, old_fact_type TEXT,
    new_kind TEXT, new_source TEXT, new_confidence REAL, new_fact_type TEXT,
    actor TEXT NOT NULL,
    changed_at TEXT NOT NULL)"""

#: The two new tables and their indexes, one statement per string.
_STATEMENTS: tuple[str, ...] = (
    _RUNS_TABLE,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_kind_run_active ON memory_kind_runs "
    "(profile_id) WHERE status IN ('queued','running','paused','reverting')",
    _HISTORY_TABLE,
    "CREATE INDEX IF NOT EXISTS idx_kind_history_run "
    "ON memory_kind_history (run_id, history_id)",
    "CREATE INDEX IF NOT EXISTS idx_kind_history_fact "
    "ON memory_kind_history (profile_id, fact_id, history_id)",
)

_NEW_TABLES = ("memory_kind_runs", "memory_kind_history")
_NEW_INDEXES = (
    "uq_kind_run_active", "idx_kind_history_run", "idx_kind_history_fact",
)


# --- helpers (row-factory tolerant; copied from M046, not imported) ----------

def _first(row):
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()), None)
    return row[0]


def _exists(conn: sqlite3.Connection, kind: str, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type=? AND name=?", (kind, name),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return _exists(conn, "table", name)


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Column names in declared order, tolerant of the connection's row factory."""
    out: list[str] = []
    try:
        for row in conn.execute(f"PRAGMA table_info({table})"):
            if isinstance(row, dict):
                out.append(str(row.get("name", "")))
            else:
                try:
                    out.append(str(row["name"]))
                except (TypeError, IndexError, KeyError):
                    out.append(str(row[1]))
    except sqlite3.Error:
        return []
    return [c for c in out if c]


def _count(conn: sqlite3.Connection, sql: str) -> int:
    try:
        value = _first(conn.execute(sql).fetchone())
    except sqlite3.Error:
        return -1
    return -1 if value is None else int(value)


# --- the migration -----------------------------------------------------------

def apply(conn: sqlite3.Connection) -> None:
    """Add the tables, then the columns and index, all in one transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        for stmt in _STATEMENTS:
            conn.execute(stmt)
        if _table_exists(conn, _TABLE):
            before = _count(conn, f"SELECT COUNT(*) FROM {_TABLE}")
            if before < 0:
                raise sqlite3.OperationalError(f"M052: cannot count {_TABLE}")
            have = set(_columns(conn, _TABLE))
            for name, sql_type in _COLUMNS:
                if name not in have:
                    conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {name} {sql_type}")
            if "profile_id" in have:
                conn.execute(_IDX_FACTS_MEMORY_KIND)
            after = _count(conn, f"SELECT COUNT(*) FROM {_TABLE}")
            if after != before:
                raise sqlite3.OperationalError(
                    f"M052: row count changed from {before} to {after}; rolling back"
                )
        conn.execute("COMMIT")
    except BaseException:
        # Any failure, including an interrupt, leaves the store as it was.
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - nothing left to roll back
            pass
        raise


def unmet(conn: sqlite3.Connection) -> str:
    """The first object this migration promises that is missing, or ''."""
    for table in _NEW_TABLES:
        if not _table_exists(conn, table):
            return f"table {table} is missing"
    for index in _NEW_INDEXES:
        if not _exists(conn, "index", index):
            return f"index {index} is missing"
    if not _table_exists(conn, _TABLE):
        return ""
    have = set(_columns(conn, _TABLE))
    for name, _sql_type in _COLUMNS:
        if name not in have:
            return f"column {_TABLE}.{name} is missing"
    if "profile_id" in have and not _exists(conn, "index", "idx_facts_memory_kind"):
        return "index idx_facts_memory_kind is missing"
    return ""


def verify(conn: sqlite3.Connection) -> bool:
    """Both tables exist; if atomic_facts exists, all five columns and the index do."""
    return unmet(conn) == ""


def repair(conn: sqlite3.Connection) -> None:
    """Re-run apply. Every statement is idempotent and nothing reads a row."""
    apply(conn)


def blocks_serving(conn: sqlite3.Connection) -> bool:
    """Never. Every kind feature checks for the columns before it writes, so a
    store this migration has not reached serves exactly as 4.1.18 did."""
    return False
