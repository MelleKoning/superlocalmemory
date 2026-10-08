# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``read_only_snapshot`` reuses a per-thread query-only connection, never a stale one.

A fresh ``mode=ro`` connection per snapshot re-parsed the schema under SQLite's
process-wide locks; recalls of 4-8 s sat at the first statement of the
correction admission's snapshot (storage/snapshot_pool.py).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from superlocalmemory.storage import snapshot_pool
from superlocalmemory.storage.read_connection import read_only_snapshot


@pytest.fixture()
def path(tmp_path: Path) -> Path:
    p = (tmp_path / "memory.db").resolve()
    c = sqlite3.connect(str(p))
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE notes (k TEXT PRIMARY KEY, v TEXT)")
    c.executemany("INSERT INTO notes VALUES (?, ?)", [("a", "one"), ("b", "two"), ("c", "three")])
    c.commit()
    c.close()
    yield p
    snapshot_pool.close_path(p)


def _write(path: Path, sql: str, params: tuple = ()) -> None:
    c = sqlite3.connect(str(path))
    c.execute(sql, params)
    c.commit()
    c.close()


def _values(path: Path) -> list[str]:
    with read_only_snapshot(path) as conn:
        return [r[0] for r in conn.execute("SELECT v FROM notes ORDER BY k")]


def test_snapshots_on_one_thread_share_one_connection(path: Path, monkeypatch) -> None:
    opened: list[int] = []
    real = snapshot_pool._open
    monkeypatch.setattr(snapshot_pool, "_open", lambda *a: opened.append(1) or real(*a))
    for _ in range(10):
        _values(path)
    assert len(opened) == 1
    assert snapshot_pool.open_count(path) == 1


def test_another_process_commit_is_seen_next_time(path: Path) -> None:
    assert _values(path) == ["one", "two", "three"]
    _write(path, "UPDATE notes SET v = 'uno' WHERE k = 'a'")
    assert _values(path)[0] == "uno"


def test_a_half_read_cursor_does_not_pin_a_stale_snapshot(path: Path) -> None:
    with read_only_snapshot(path) as conn:
        cur = conn.execute("SELECT v FROM notes ORDER BY k")
        assert cur.fetchone()[0] == "one"  # left half-read on purpose
        kept = cur
    _write(path, "INSERT INTO notes VALUES ('d', 'four')")
    assert _values(path) == ["one", "two", "three", "four"]
    with pytest.raises(sqlite3.ProgrammingError):
        kept.fetchone()  # closed at the block's end, as a closed connection was


def test_a_schema_change_is_seen(path: Path) -> None:
    _values(path)
    _write(path, "CREATE TABLE later (n INTEGER)")
    _write(path, "INSERT INTO later VALUES (5)")
    with read_only_snapshot(path) as conn:
        assert conn.execute("SELECT n FROM later").fetchone()[0] == 5


def test_a_replaced_file_is_reopened(path: Path, tmp_path: Path) -> None:
    _values(path)
    staged = tmp_path / "staged.db"
    shutil.copy(path, staged)
    _write(staged, "UPDATE notes SET v = 'restored'")
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    os.replace(staged, path)
    assert set(_values(path)) == {"restored"}


def test_caller_settings_do_not_leak_into_the_next_snapshot(path: Path) -> None:
    with read_only_snapshot(path) as conn:
        conn.set_progress_handler(lambda: 1, 1)  # would abort every later statement
        conn.row_factory = None
    with read_only_snapshot(path) as conn:
        row = conn.execute("SELECT v FROM notes WHERE k = 'a'").fetchone()
        assert row["v"] == "one"  # Row factory restored, no handler aborting it


def test_it_stays_query_only(path: Path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        with read_only_snapshot(path) as conn:
            conn.execute("INSERT INTO notes VALUES ('x', 'y')")
    assert _values(path) == ["one", "two", "three"]


def test_an_error_in_the_block_drops_the_connection(path: Path) -> None:
    _values(path)
    with pytest.raises(RuntimeError):
        with read_only_snapshot(path):
            raise RuntimeError("caller failed")
    assert snapshot_pool.open_count(path) == 0
    assert _values(path) == ["one", "two", "three"]


def test_close_reaches_every_thread_and_the_file_can_move(path: Path) -> None:
    ready, done = threading.Barrier(2), threading.Event()

    def worker():
        _values(path)
        ready.wait()
        done.wait(timeout=30)

    t = threading.Thread(target=worker)
    t.start()
    ready.wait()
    _values(path)
    assert snapshot_pool.open_count(path) == 2
    snapshot_pool.close_path(path)
    assert snapshot_pool.open_count(path) == 0
    done.set()
    t.join()
    moved = path.with_name("moved.db")
    os.rename(path, moved)
    os.rename(moved, path)


def test_only_a_few_files_are_held(tmp_path: Path) -> None:
    paths = []
    for i in range(snapshot_pool.MAX_PATHS + 3):
        p = (tmp_path / f"s{i}.db").resolve()
        c = sqlite3.connect(str(p))
        c.execute("CREATE TABLE t (x)")
        c.commit()
        c.close()
        with read_only_snapshot(p) as conn:
            conn.execute("SELECT * FROM t").fetchall()
        paths.append(p)
    assert sum(snapshot_pool.open_count(p) for p in paths) == snapshot_pool.MAX_PATHS
    for p in paths:
        snapshot_pool.close_path(p)


def test_an_erased_row_is_gone_on_the_reused_connection(path: Path) -> None:
    assert "two" in _values(path)
    _write(path, "DELETE FROM notes WHERE k = 'b'")
    assert _values(path) == ["one", "three"]


def test_a_non_default_timeout_keeps_its_own_fresh_connection(path: Path) -> None:
    with read_only_snapshot(path, timeout_ms=250) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 3
    assert snapshot_pool.open_count(path) == 0
