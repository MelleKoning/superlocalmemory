# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A plain read reuses its thread's connection, and that never makes an answer stale.

A fresh connection per statement re-parsed the schema and rebuilt a page cache,
under SQLite's process-wide allocation and page-cache locks: 4.3 ms per
statement alone, 23.7 ms with six threads, against 0.04 / 0.15 ms reused
(storage/read_connection_pool.py). These tests pin what reuse must never
change: freshness, isolation from transactions, cleanup.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from superlocalmemory.storage import read_connection_pool as rcp
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.execute("CREATE TABLE notes (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO notes VALUES ('a', 'one')")
    conn.commit()
    conn.close()
    manager = DatabaseManager(path)
    yield manager
    manager.close()


def _current(db: DatabaseManager) -> sqlite3.Connection | None:
    pool = getattr(db, "_read_pool", None)
    entry = getattr(pool._local, "entry", None) if pool is not None else None
    return None if entry is None or entry.closed else entry.conn


def _read(db: DatabaseManager, sql: str, params: tuple = ()) -> list:
    rows = db.execute(sql, params)
    conn = _current(db)
    assert conn is None or conn.in_transaction is False, "a reused connection kept a snapshot"
    return [tuple(r) for r in rows]


def test_reads_on_one_thread_share_one_connection(db: DatabaseManager, monkeypatch) -> None:
    from superlocalmemory.storage import database

    opened: list[int] = []
    real = database.sqlite3.connect
    monkeypatch.setattr(database.sqlite3, "connect", lambda *a, **k: opened.append(1) or real(*a, **k))
    for _ in range(20):
        _read(db, "SELECT v FROM notes WHERE k = ?", ("a",))
    assert len(opened) == 1, f"{len(opened)} connections opened for 20 reads on one thread"


def test_a_commit_from_another_process_is_seen_on_the_next_read(db: DatabaseManager) -> None:
    assert _read(db, "SELECT v FROM notes ORDER BY k") == [("one",)]
    other = sqlite3.connect(str(db.db_path))
    other.execute("INSERT INTO notes VALUES ('b', 'two')")
    other.commit()
    other.close()
    assert _read(db, "SELECT v FROM notes ORDER BY k") == [("one",), ("two",)]


def test_a_schema_change_by_another_connection_is_seen(db: DatabaseManager) -> None:
    _read(db, "SELECT v FROM notes")
    other = sqlite3.connect(str(db.db_path))
    other.execute("ALTER TABLE notes ADD COLUMN extra TEXT DEFAULT 'x'")
    other.execute("CREATE TABLE later (n INTEGER)")
    other.execute("INSERT INTO later VALUES (7)")
    other.commit()
    other.close()
    assert _read(db, "SELECT extra FROM notes") == [("x",)]
    assert _read(db, "SELECT n FROM later") == [(7,)]


def test_a_replaced_store_file_is_reopened(db: DatabaseManager, tmp_path: Path) -> None:
    _read(db, "SELECT v FROM notes")
    old = _current(db)
    staged = tmp_path / "staged.db"
    shutil.copy(db.db_path, staged)
    s = sqlite3.connect(str(staged))
    s.execute("UPDATE notes SET v = 'restored'")
    s.commit()
    s.close()
    for suffix in ("-wal", "-shm"):
        Path(f"{db.db_path}{suffix}").unlink(missing_ok=True)
    os.replace(staged, db.db_path)  # what a restore does
    assert _read(db, "SELECT v FROM notes") == [("restored",)]
    assert _current(db) is not old
    with pytest.raises(sqlite3.ProgrammingError):
        old.execute("SELECT 1")  # the old one was closed, not leaked


def test_a_read_inside_a_transaction_uses_the_transaction(db: DatabaseManager) -> None:
    with db.transaction():
        db.execute("INSERT INTO notes VALUES ('c', 'three')")
        assert [tuple(r) for r in db.execute("SELECT v FROM notes WHERE k='c'")] == [("three",)]
        assert _current(db) is None, "a transaction's read went to the reused connection"
    with db.raw_connection() as conn:
        conn.execute("INSERT INTO notes VALUES ('d', 'four')")
        assert [tuple(r) for r in db.execute("SELECT v FROM notes WHERE k='d'")] == [("four",)]
        assert _current(db) is None


def test_an_error_drops_the_connection(db: DatabaseManager) -> None:
    _read(db, "SELECT v FROM notes")
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT nothing FROM missing_table")
    assert _current(db) is None and db._read_pool.open_count() == 0
    assert _read(db, "SELECT v FROM notes") == [("one",)]


def test_writes_and_pragmas_never_use_the_reused_connection(db: DatabaseManager) -> None:
    db.execute("INSERT INTO notes VALUES ('e', 'five')")
    db.execute("PRAGMA table_info(notes)")
    assert db._read_pool.open_count() == 0


def test_close_reaches_every_thread_and_leaves_nothing_open(db: DatabaseManager) -> None:
    ready, done = threading.Barrier(3), threading.Event()
    conns: list[sqlite3.Connection] = []

    def worker():
        db.execute("SELECT v FROM notes")
        conns.append(_current(db))
        ready.wait()
        done.wait(timeout=30)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    ready.wait()
    db.execute("SELECT v FROM notes")
    conns.append(_current(db))
    assert db._read_pool.open_count() == 3
    db.close()
    assert db._read_pool.open_count() == 0
    for conn in conns:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    done.set()
    for t in threads:
        t.join()
    moved = db.db_path.with_name("moved.db")
    os.rename(db.db_path, moved)  # nothing holds it (a Windows handle would refuse)
    os.rename(moved, db.db_path)


def test_a_finished_thread_gives_its_connection_back(db: DatabaseManager) -> None:
    t = threading.Thread(target=lambda: db.execute("SELECT v FROM notes"))
    t.start()
    t.join()
    assert db._read_pool.open_count() == 1
    db.execute("SELECT v FROM notes")  # the next checkout closes the dead thread's
    assert db._read_pool.open_count() == 1


def test_beyond_the_cap_a_read_opens_its_own(db: DatabaseManager, monkeypatch) -> None:
    db._read_pool._max = 1
    db.execute("SELECT v FROM notes")
    out: list = []
    t = threading.Thread(target=lambda: out.append(
        ([tuple(r) for r in db.execute("SELECT v FROM notes")], _current(db))))
    t.start()
    t.join()
    assert out == [([("one",)], None)]
    assert db._read_pool.open_count() == 1


def test_an_erased_memory_is_gone_on_the_same_connection(db: DatabaseManager) -> None:
    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1','default','s')")
    db.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
               "VALUES ('f1','m1','default','private text')")
    assert _read(db, "SELECT fact_id FROM atomic_facts WHERE fact_id='f1'") == [("f1",)]
    reused = _current(db)
    other = sqlite3.connect(str(db.db_path))
    other.execute("PRAGMA foreign_keys=OFF")
    other.execute("DELETE FROM atomic_facts WHERE fact_id='f1'")
    other.commit()
    other.close()
    assert _read(db, "SELECT fact_id FROM atomic_facts WHERE fact_id='f1'") == []
    assert _current(db) is reused


def test_only_plain_reads_qualify() -> None:
    assert rcp.is_plain_read("  select 1") and rcp.is_plain_read("WITH x AS (SELECT 1) SELECT * FROM x")
    for sql in ("INSERT INTO t VALUES (1)", "PRAGMA table_info(t)", "VACUUM", "ATTACH 'x' AS y",
                "BEGIN", "EXPLAIN QUERY PLAN SELECT 1"):
        assert not rcp.is_plain_read(sql), sql
