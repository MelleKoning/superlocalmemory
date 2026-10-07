# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
""""Is this memory visible?" is answered from an index, not from the row.

The visibility columns (``archive_status``, ``quarantined``) sit after the
embedding and both Fisher vectors, ~40 KB into each row, so every visibility
check walked the row's overflow chain. On a cold 22k-fact store the visible
count took 5.5 s and a recall's authorization reads took seconds while the
daemon warmed up; with a covering index they are a few milliseconds.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.storage.database import DatabaseManager, _scope_where
from superlocalmemory.storage.schema import create_all_tables


def _plan(db: DatabaseManager, sql: str, params: tuple) -> str:
    return " | ".join(str(dict(r)["detail"]) for r in db.execute("EXPLAIN QUERY PLAN " + sql, params))


def _upgraded_store(tmp_path: Path) -> DatabaseManager:
    """A store as an upgrade finds it: tables exist, visibility columns arrived later."""
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.execute("ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live'")
    conn.execute("DROP INDEX IF EXISTS idx_facts_visibility")
    conn.commit()
    create_all_tables(conn)  # the next engine start
    conn.commit()
    conn.close()
    return DatabaseManager(path)


def test_the_visible_count_reads_only_an_index(tmp_path: Path) -> None:
    db = _upgraded_store(tmp_path)
    where, params = _scope_where("default")
    plan = _plan(db, f"SELECT COUNT(*) AS c FROM atomic_facts WHERE {where}{db.visible_fact_clause()}",
                 tuple(params))
    assert "COVERING INDEX" in plan, plan


def test_which_of_these_ids_are_visible_reads_only_an_index(tmp_path: Path) -> None:
    db = _upgraded_store(tmp_path)
    where, params = _scope_where("default")
    plan = _plan(db, "SELECT fact_id FROM atomic_facts WHERE fact_id IN (?,?) "
                     f"AND {where}{db.visible_fact_clause()}", ("a", "b", *params))
    assert "COVERING INDEX" in plan, plan


def test_a_store_without_the_columns_still_starts(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "old.db"))
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT)")
    from superlocalmemory.storage import visibility_index

    visibility_index.ensure(conn)  # no archive_status: nothing to index, no error
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")]
    assert "idx_facts_visibility" not in names
