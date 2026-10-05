# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M054 — saved views, in ``learning.db`` (issue #113).

WHAT IT ADDS
------------
``saved_views``: one row per named, profile-scoped recall query. A view is a
name, the query text, a small validated set of recall filters (``filters_json``)
and how many results to show. Running one calls the normal recall path, so it
ranks, checks and repeats exactly as that query would. Nothing here stores a
result: what a view shows is decided by recall each time it runs.

``name_key`` is the case-folded name, so "Work log" and "work log" cannot both
exist in one profile (``idx_saved_views_profile_name``). Uniqueness is per
profile: two profiles may each have a view called "Work log".

WHY learning.db
---------------
A view is a person's setting, not a memory. Keeping it out of memory.db means
this release's only migration changes the smaller database, so the safety copy
taken on upgrade is of learning.db alone and never of memory.db (see
``tests/test_storage/test_an_update_copies_only_what_it_changes.py``).

NO CHECK CONSTRAINTS
--------------------
Values are validated before they are written (``views.model``). A CHECK would
turn a validation change in a later release into a failed write on old rows.

OLDER BUILDS
------------
Additive: no earlier build names this table, so ``BREAKING_VERSION`` is 0.
``DOWNGRADE_FLOOR`` is 51: 4.1.20 (schema 53), 4.1.19 (52) and 4.1.18 (51) all
ignore a table they do not know, so ``slm db prepare-downgrade`` — which stamps
the store 51 so that any of those releases opens it — keeps working exactly as
M052 and M053 left it. A floor of 53 would refuse that path outright.
"""

from __future__ import annotations

import sqlite3

NAME = "M054_saved_views"
DB_TARGET = "learning"

#: Additive: a build that predates this never reads or writes what it adds.
BREAKING_VERSION = 0

#: ``prepare_downgrade`` may lower the recorded schema version to this.
DOWNGRADE_FLOOR = 51

DDL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS saved_views (
    profile_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    query TEXT NOT NULL,
    filters_json TEXT NOT NULL DEFAULT '{}',
    result_limit INTEGER NOT NULL DEFAULT 10,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, view_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_views_profile_name
    ON saved_views (profile_id, name_key);
COMMIT;
"""

TABLE = "saved_views"
_INDEX = "idx_saved_views_profile_name"
_COLUMNS = frozenset({
    "profile_id", "view_id", "name", "name_key", "query", "filters_json",
    "result_limit", "created_at", "updated_at",
})


def verify(conn: sqlite3.Connection) -> bool:
    """The table, every column and the per-profile name index exist."""
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}
    if TABLE not in names or _INDEX not in names:
        return False
    have = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE})")}
    return _COLUMNS <= have


def repair(conn: sqlite3.Connection) -> None:
    """Re-create whatever is missing. Every statement is IF NOT EXISTS."""
    conn.executescript(DDL)
    if not verify(conn):
        raise sqlite3.OperationalError("M054 schema did not reach its required end-state")


def blocks_serving(conn: sqlite3.Connection) -> bool:
    """Never: without the table, saved views say they are unavailable."""
    return False
