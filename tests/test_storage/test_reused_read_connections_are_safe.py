"""Reused read connections: writes never slip onto them, closing them is bounded,
and a forked child never uses its parent's connection.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from superlocalmemory.storage import read_connection_pool as rcp
from superlocalmemory.storage.database import DatabaseManager


def _manager(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (id, v) VALUES (?, ?)", [(i, f"v{i}") for i in range(5)])
    conn.commit()
    conn.close()
    return DatabaseManager(path)


# -- a write introduced by WITH is a write -------------------------------------

@pytest.mark.parametrize("sql, read", [
    ("SELECT 1", True),
    ("  with x AS (SELECT 1) SELECT * FROM x", True),
    ("WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<3) SELECT n FROM c",
     True),
    ("WITH x AS (SELECT 1) VALUES (1)", True),
    ("WITH m AS (SELECT id FROM t WHERE v = 'a)') DELETE FROM t WHERE id IN m", False),
    ("WITH m AS (SELECT 1) INSERT INTO t (v) SELECT 'x' FROM m", False),
    ("with m as materialized (select 1) update t set v = 'y'", False),
    ("WITH m AS (SELECT 1) /* SELECT */ REPLACE INTO t (id, v) VALUES (1, 'z')", False),
    ('WITH "select" AS (SELECT 1) DELETE FROM t', False),
    ("WITH m AS (SELECT 1", False),
    ("WITHOUT", False),
    ("UPDATE t SET v = 'q'", False),
])
def test_statements_are_classified_by_their_main_statement(sql: str, read: bool) -> None:
    assert rcp.is_plain_read(sql) is read


class _SpyLock:
    def __init__(self, real) -> None:
        self.real, self.entered = real, 0

    def __enter__(self):
        self.entered += 1
        return self.real.__enter__()

    def __exit__(self, *exc):
        return self.real.__exit__(*exc)


def test_a_with_delete_takes_the_single_writer_lock(tmp_path: Path, monkeypatch) -> None:
    db = _manager(tmp_path)
    spy = _SpyLock(db._lock)
    monkeypatch.setattr(db, "_lock", spy)
    db.execute("SELECT COUNT(*) FROM t")  # a plain read takes no lock
    assert spy.entered == 0
    db.execute("WITH m AS (SELECT id FROM t WHERE id < 2) DELETE FROM t WHERE id IN m")
    assert spy.entered == 1
    assert dict(db.execute("SELECT COUNT(*) AS n FROM t")[0])["n"] == 3


def test_a_reused_read_connection_refuses_writes(tmp_path: Path) -> None:
    db = _manager(tmp_path)
    db.execute("SELECT 1")
    conn = db._read_pool.checkout()
    assert conn is not None
    with pytest.raises(sqlite3.OperationalError, match="readonly|read-only|query"):
        conn.execute("DELETE FROM t")
    assert dict(db.execute("SELECT COUNT(*) AS n FROM t")[0])["n"] == 5


# -- closing is bounded, and nothing is leaked ---------------------------------

def _blocking_pool(path: Path, gate: threading.Event, entered: threading.Semaphore):
    def opener() -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), check_same_thread=False)

        def block() -> int:
            entered.release()
            gate.wait(30)
            return 1

        conn.create_function("block", 0, block)
        return conn

    return rcp.ReadConnectionPool(path, opener)


def _fresh(_sql, _params):
    raise AssertionError("the pooled connection should have been used")


def test_close_waits_one_bounded_time_for_all_busy_connections(tmp_path: Path,
                                                               monkeypatch) -> None:
    db = _manager(tmp_path)
    gate, entered = threading.Event(), threading.Semaphore(0)
    pool = _blocking_pool(db.db_path, gate, entered)
    conns: list[sqlite3.Connection] = []

    def reader() -> None:
        conns.append(pool.checkout())
        rcp.execute_read(pool, "SELECT block()", (), _fresh, retries=1, base_delay=0.01)

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for thread in threads:
        thread.start()
    for _ in threads:
        assert entered.acquire(timeout=10)
    monkeypatch.setattr(rcp, "CLOSE_WAIT_SECONDS", 0.5)
    try:
        started = time.monotonic()
        pool.close_all()
        took = time.monotonic() - started
    finally:
        gate.set()
        for thread in threads:
            thread.join(10)
    # One shared deadline: per-connection waits would take 6 x 0.5 s.
    assert took < 1.5, f"close took {took:.2f}s for 6 busy connections"
    # Each connection still busy at the deadline was closed by its own thread.
    for conn in conns:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_a_snapshot_closed_while_in_use_is_closed_by_its_thread(tmp_path: Path,
                                                                monkeypatch) -> None:
    from superlocalmemory.storage import snapshot_pool

    db = _manager(tmp_path)
    path = Path(db.db_path).resolve()
    monkeypatch.setattr(rcp, "CLOSE_WAIT_SECONDS", 0.2)
    inside, leave = threading.Event(), threading.Event()
    held: list = []

    @contextmanager
    def fresh():
        raise AssertionError("the pooled connection should have been used")
        yield  # pragma: no cover

    def reader() -> None:
        with snapshot_pool.snapshot(path, snapshot_pool.DEFAULT_TIMEOUT_MS, fresh) as conn:
            held.append(conn._conn)
            inside.set()
            leave.wait(10)

    thread = threading.Thread(target=reader)
    thread.start()
    assert inside.wait(10)
    snapshot_pool.close_path(path)  # gives up waiting after 0.2 s
    leave.set()
    thread.join(10)
    with pytest.raises(sqlite3.ProgrammingError):
        held[0].execute("SELECT 1")


# -- a forked child never uses its parent's connection -------------------------

@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
def test_a_forked_child_opens_its_own_read_connection(tmp_path: Path) -> None:
    db = _manager(tmp_path)
    db.execute("SELECT 1")
    parent_conn = db._read_pool.checkout()
    assert parent_conn is not None
    read_fd, write_fd = os.pipe()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork in a threaded runner
        pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        code = 1
        try:
            child_conn = db._read_pool.checkout()
            rows = db.execute("SELECT COUNT(*) AS n FROM t")
            same = child_conn is parent_conn
            os.write(write_fd, f"{int(same)} {dict(rows[0])['n']}".encode())
            code = 0
        finally:
            os._exit(code)
    os.close(write_fd)
    _, status = os.waitpid(pid, 0)
    answer = os.read(read_fd, 64).decode()
    os.close(read_fd)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, sys.stderr
    same, count = answer.split()
    assert same == "0", "the child reused the parent's pooled connection"
    assert count == "5"
    # The parent's connection is untouched by the child.
    assert parent_conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 5


def test_an_entry_from_another_process_is_never_handed_out(tmp_path: Path) -> None:
    """Without the at-fork reset (a platform lacking it), the pid check still holds."""
    db = _manager(tmp_path)
    db.execute("SELECT 1")
    entry = db._read_pool._local.entry
    entry.pid = os.getpid() + 1  # as if opened by the parent before a fork
    fresh = db._read_pool.checkout_entry()
    assert fresh is not None and fresh is not entry
    inherited = getattr(rcp, "_INHERITED", [])
    assert entry in inherited  # kept, never closed in the wrong process
    inherited.remove(entry)
