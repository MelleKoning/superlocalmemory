# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Stores shaped like a 4.1.18 memory.db, for the memory-kind migration tests.

Two builders:

``reduced_store`` is a faithful reduction of the shipped table — TEXT primary
key (so rowid is implicit and the search index depends on it), the fact_type
CHECK, the external-content FTS5 index with its three triggers, and the two
``memory_scenes`` triggers that name ``atomic_facts`` from another table.

``full_pre_kind_store`` builds the real current schema and then removes the
memory-kind columns, which yields exactly what a 4.1.18 engine created. It needs
``ALTER TABLE ... DROP COLUMN`` (SQLite 3.35+).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

KIND_COLUMNS = (
    "memory_kind", "memory_kind_source", "memory_kind_confidence",
    "memory_kind_recipe", "memory_kind_at",
)

_REDUCED_DDL = """
CREATE TABLE atomic_facts (
    fact_id     TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    profile_id  TEXT NOT NULL DEFAULT 'default',
    content     TEXT NOT NULL,
    fact_type   TEXT NOT NULL DEFAULT 'semantic'
                    CHECK (fact_type IN (
                        'episodic', 'semantic', 'opinion', 'prospective'
                    )),
    quarantined INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_facts_type ON atomic_facts (profile_id, fact_type);
CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(
    fact_id UNINDEXED, content,
    content='atomic_facts', content_rowid='rowid'
);
CREATE TRIGGER atomic_facts_fts_insert AFTER INSERT ON atomic_facts BEGIN
    INSERT INTO atomic_facts_fts (rowid, fact_id, content)
    VALUES (NEW.rowid, NEW.fact_id, NEW.content);
END;
CREATE TRIGGER atomic_facts_fts_delete AFTER DELETE ON atomic_facts BEGIN
    INSERT INTO atomic_facts_fts (atomic_facts_fts, rowid, fact_id, content)
    VALUES ('delete', OLD.rowid, OLD.fact_id, OLD.content);
END;
CREATE TRIGGER atomic_facts_fts_update AFTER UPDATE OF content ON atomic_facts
BEGIN
    INSERT INTO atomic_facts_fts (atomic_facts_fts, rowid, fact_id, content)
    VALUES ('delete', OLD.rowid, OLD.fact_id, OLD.content);
    INSERT INTO atomic_facts_fts (rowid, fact_id, content)
    VALUES (NEW.rowid, NEW.fact_id, NEW.content);
END;
CREATE TABLE memory_scenes (
    scene_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL DEFAULT 'default',
    fact_ids_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE scene_fact_members (
    profile_id TEXT, scene_id TEXT, fact_id TEXT, position INTEGER,
    PRIMARY KEY (scene_id, fact_id));
CREATE TRIGGER trg_scene_fact_members_insert AFTER INSERT ON memory_scenes
BEGIN
    DELETE FROM scene_fact_members WHERE scene_id = NEW.scene_id;
    INSERT OR IGNORE INTO scene_fact_members (profile_id, scene_id, fact_id, position)
    SELECT NEW.profile_id, NEW.scene_id, af.fact_id, CAST(member.key AS INTEGER)
    FROM json_each(CASE WHEN json_valid(NEW.fact_ids_json)
                        THEN NEW.fact_ids_json ELSE '[]' END) AS member
    JOIN atomic_facts AS af
      ON af.fact_id = member.value AND af.profile_id = NEW.profile_id;
END;
CREATE TRIGGER trg_scene_fact_members_update
AFTER UPDATE OF profile_id, fact_ids_json ON memory_scenes
BEGIN
    DELETE FROM scene_fact_members WHERE scene_id = NEW.scene_id;
    INSERT OR IGNORE INTO scene_fact_members (profile_id, scene_id, fact_id, position)
    SELECT NEW.profile_id, NEW.scene_id, af.fact_id, CAST(member.key AS INTEGER)
    FROM json_each(CASE WHEN json_valid(NEW.fact_ids_json)
                        THEN NEW.fact_ids_json ELSE '[]' END) AS member
    JOIN atomic_facts AS af
      ON af.fact_id = member.value AND af.profile_id = NEW.profile_id;
END;
"""

ROWS = [
    ("f1", "m1", "default", "Dentist appointment on the fourth", "prospective"),
    ("f2", "m1", "default", "Paris is the capital of France", "semantic"),
    ("f3", "m2", "other", "Went to the market on Tuesday", "episodic"),
    ("f4", "m2", "default", "I think the new pipeline is faster", "opinion"),
    ("f5", "m3", "default", "The release pipeline uses signed wheels", "semantic"),
    ("f6", "m3", "shared", "Quarterly review scheduled for March", "prospective"),
]


def connect(path: Path) -> sqlite3.Connection:
    """A connection shaped like the migration runner's (autocommit)."""
    return sqlite3.connect(str(path), isolation_level=None)


def reduced_store(path: Path) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(_REDUCED_DDL)
    # Delete one row so rowids have a gap: a copy that re-assigned rowids
    # would visibly drift from the search index.
    conn.executemany(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, fact_type) "
        "VALUES (?,?,?,?,?)",
        [("gap", "m0", "default", "deleted before migration", "semantic"), *ROWS],
    )
    conn.execute("DELETE FROM atomic_facts WHERE fact_id='gap'")
    conn.execute("INSERT INTO memory_scenes VALUES ('s1','default','[\"f1\",\"f2\"]')")
    return conn


def full_pre_kind_store(path: Path) -> sqlite3.Connection:
    """The real schema as a 4.1.18 engine created it (no kind columns)."""
    if sqlite3.sqlite_version_info < (3, 35, 0):
        pytest.skip("DROP COLUMN needs SQLite 3.35+")
    from superlocalmemory.storage import schema

    conn = connect(path)
    schema.create_all_tables(conn)
    conn.execute("DROP INDEX IF EXISTS idx_facts_memory_kind")
    have = columns(conn)
    for name in KIND_COLUMNS:
        if name in have:
            conn.execute(f"ALTER TABLE atomic_facts DROP COLUMN {name}")
    assert not set(KIND_COLUMNS) & set(columns(conn))
    return conn


def columns(conn: sqlite3.Connection, table: str = "atomic_facts") -> list[str]:
    return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")]


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def object_names(conn: sqlite3.Connection, kind: str) -> set[str]:
    return {
        str(r[0]) for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type=?", (kind,))
    }


def fts_ok(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute(
            "INSERT INTO atomic_facts_fts(atomic_facts_fts) VALUES('integrity-check')"
        )
        return True
    except sqlite3.DatabaseError:
        return False


class FailOnNthAlter:
    """Wraps a connection; raises on the n-th ALTER, refuses executescript.

    ``executescript`` is refused because it COMMITs any open transaction, which
    would make a rollback-on-failure test pass for the wrong reason.
    """

    def __init__(self, conn: sqlite3.Connection, n: int) -> None:
        self._conn = conn
        self._n = n
        self.alters = 0

    def execute(self, sql: str, *args):
        if sql.lstrip().upper().startswith("ALTER TABLE"):
            self.alters += 1
            if self.alters == self._n:
                raise sqlite3.OperationalError("injected failure on ALTER")
        return self._conn.execute(sql, *args)

    def executescript(self, *_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("executescript commits the open transaction")

    def __getattr__(self, name):
        return getattr(self._conn, name)
