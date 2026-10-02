# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""A restore into a live store does not leave a write-ahead log the size of the store.

Audit M-14, reproduced before the fix: a 39 MB store restored into a WAL store
held open by another connection left its ``-wal`` at 39.4 MB after one more
commit, and nothing ever shrank it -- about 2 GB of disk per restore on a 2 GB
store, for ever.

Real temp directories and real SQLite files only.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from superlocalmemory.storage.backup import (
    LiveStoreWriteError,
    _write_into_live_db,
    restore_pre_migration_snapshot,
)

_BODY = "y" * 1000
_ROWS = 6000  # ~6 MB of pages


def _store(path: Path, tag: str, rows: int) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, tag TEXT, body TEXT)")
        conn.executemany("INSERT INTO t (tag, body) VALUES (?, ?)", [(tag, _BODY)] * rows)
        conn.commit()
    finally:
        conn.close()


def _wal_size(db: Path) -> int:
    wal = Path(f"{db}-wal")
    return wal.stat().st_size if wal.exists() else 0


def _daemon(db: Path) -> sqlite3.Connection:
    """A long-lived connection, as the daemon holds one."""
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.execute("SELECT count(*) FROM t").fetchone()
    return conn


@pytest.fixture
def stores(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "snap.db"
    target = tmp_path / "memory.db"
    _store(source, "restored", _ROWS)
    conn = sqlite3.connect(str(source))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    _store(target, "live", 50)
    return source, target


def test_a_restore_into_a_live_store_leaves_a_small_log(stores) -> None:
    source, target = stores
    daemon = _daemon(target)
    try:
        _write_into_live_db(source, target)
        daemon.execute("INSERT INTO t (tag, body) VALUES ('after', 'z')")
        daemon.commit()
        size = _wal_size(target)
        assert dict(daemon.execute("SELECT tag, count(*) FROM t GROUP BY tag")) == {
            "after": 1, "restored": _ROWS,
        }
    finally:
        daemon.close()
    store = target.stat().st_size
    assert size < store / 10, f"-wal is {size:,} bytes beside a {store:,}-byte store"


def test_a_pre_migration_restore_leaves_a_small_log_too(stores, tmp_path) -> None:
    source, target = stores
    daemon = _daemon(target)
    try:
        restore_pre_migration_snapshot(source, target)
        size = _wal_size(target)
    finally:
        daemon.close()
    assert size < target.stat().st_size / 10


def test_a_reader_mid_query_is_never_disturbed(stores) -> None:
    """The log is only shrunk once no reader needs it; a reader that is still
    inside its snapshot keeps reading that snapshot, consistently."""
    source, target = stores
    reader = sqlite3.connect(str(target), isolation_level=None, check_same_thread=False)
    reader.execute("BEGIN")
    before = reader.execute("SELECT count(*) FROM t").fetchone()[0]
    try:
        _write_into_live_db(source, target)
        # still inside its read transaction: same snapshot, no error
        assert reader.execute("SELECT count(*) FROM t").fetchone()[0] == before
        assert reader.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        reader.execute("COMMIT")
        assert reader.execute("SELECT count(*) FROM t").fetchone()[0] == _ROWS
    finally:
        reader.close()
    check = sqlite3.connect(str(target))
    try:
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        check.close()


def test_readers_running_throughout_a_restore_see_one_store_or_the_other(stores) -> None:
    source, target = stores
    seen: set[int] = set()
    errors: list[BaseException] = []
    stop = threading.Event()

    def read_loop() -> None:
        conn = sqlite3.connect(str(target), timeout=5)
        try:
            while not stop.is_set():
                seen.add(conn.execute("SELECT count(*) FROM t").fetchone()[0])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=read_loop) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        _write_into_live_db(source, target)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
    assert not errors, errors
    assert seen <= {50, _ROWS}, seen


def test_a_failed_write_also_leaves_a_small_log(stores, monkeypatch) -> None:
    """A write that fails part-way is rolled back by SQLite; the pages it had
    already written to the log must not stay on disk either."""
    source, target = stores
    daemon = _daemon(target)

    real_connect = sqlite3.connect

    class _FailingBackup:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def backup(self, dst, *, pages=-1, progress=None, **kw):
            # Write real pages into the target's log, then fail, as a lock
            # timeout part-way through does.
            dst.execute("PRAGMA cache_size=10")  # spill pages into the log
            dst.execute("BEGIN IMMEDIATE")
            dst.executemany("INSERT INTO t (tag, body) VALUES ('partial', ?)",
                            [(_BODY,)] * 4000)
            dst.execute("ROLLBACK")
            raise LiveStoreWriteError("stayed locked by another connection")

    def fake_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if "immutable=1" in str(database):
            return _FailingBackup(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    try:
        with pytest.raises(LiveStoreWriteError):
            _write_into_live_db(source, target)
        monkeypatch.setattr(sqlite3, "connect", real_connect)
        daemon.execute("INSERT INTO t (tag, body) VALUES ('after', 'z')")
        daemon.commit()
        size = _wal_size(target)
        assert dict(daemon.execute("SELECT tag, count(*) FROM t GROUP BY tag")) == {
            "after": 1, "live": 50,
        }
    finally:
        monkeypatch.setattr(sqlite3, "connect", real_connect)
        daemon.close()
    assert size < 1_000_000, f"-wal kept {size:,} bytes after a rolled-back write"
