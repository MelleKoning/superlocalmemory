# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``read_only_snapshot`` on a reused, per-thread, query-only connection.

WHY
---
Every ``read_only_snapshot`` opened a fresh ``mode=ro`` connection: the schema
(405 objects) parsed again and a page cache built, all under SQLite's
process-wide allocation and page-cache locks (storage/read_connection_pool.py
has the measurements). The recall's correction admission takes one such
snapshot per recall; sampled on a copy of a 22k-fact store, recalls of 4-8 s
sat at its first statement while the materializer wrote.

WHAT THIS DOES
--------------
The same discipline as ``DatabaseManager``'s reused reads, on query-only
connections, one per thread per store file (at most ``MAX_PATHS`` files):

* every cursor opened through the snapshot is closed when the block ends, so
  the connection never keeps a read transaction (a pinned snapshot would serve
  stale rows and starve WAL checkpoints); an open transaction is rolled back;
* a progress handler or row factory a caller set is reset on exit;
* any exception inside the block closes the connection instead of reusing it;
* a replaced store file (new inode) is reopened; past the per-file thread cap
  or with a non-default timeout the old fresh-connection path is used;
* ``close_path`` (``DatabaseManager.close``) and ``close_all`` close them all.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from superlocalmemory.storage import read_connection_pool as _rcp
from superlocalmemory.storage.read_connection_pool import ReadConnectionPool, finish_entry

#: Store files with pooled snapshot connections at once (tests open thousands).
MAX_PATHS = 8
DEFAULT_TIMEOUT_MS = 5_000

_pools: "OrderedDict[str, ReadConnectionPool]" = OrderedDict()
_pools_lock = threading.Lock()


def _open(path: Path, timeout_ms: int) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True,
                           timeout=timeout_ms / 1000.0, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={timeout_ms}")
    except BaseException:
        conn.close()
        raise
    return conn


def _pool_for(path: Path) -> ReadConnectionPool:
    key = str(path)
    evicted: list[ReadConnectionPool] = []
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            pool = ReadConnectionPool(path, lambda: _open(path, DEFAULT_TIMEOUT_MS))
            _pools[key] = pool
            while len(_pools) > MAX_PATHS:
                evicted.append(_pools.popitem(last=False)[1])
        else:
            _pools.move_to_end(key)
    for old in evicted:
        old.close_all()
    return pool


class _Tracked:
    """The connection, recording every cursor so the block's end can close them."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cursors: list[weakref.ref] = []

    def _track(self, cursor: sqlite3.Cursor) -> sqlite3.Cursor:
        self._cursors.append(weakref.ref(cursor))
        return cursor

    def execute(self, *args: Any) -> sqlite3.Cursor:
        return self._track(self._conn.execute(*args))

    def executemany(self, *args: Any) -> sqlite3.Cursor:
        return self._track(self._conn.executemany(*args))

    def cursor(self, *args: Any) -> sqlite3.Cursor:
        return self._track(self._conn.cursor(*args))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)

    def _finish(self) -> None:
        for ref in self._cursors:
            cur = ref()
            if cur is not None:
                cur.close()  # resets its statement: no read transaction survives
        self._cursors.clear()
        conn = self._conn
        conn.set_progress_handler(None, 0)
        conn.row_factory = sqlite3.Row
        if conn.in_transaction:
            conn.rollback()


@contextmanager
def snapshot(path: Path, timeout_ms: int, fresh) -> Iterator[Any]:
    """A pooled query-only connection for one block; ``fresh()`` when none is available."""
    pool = _pool_for(path) if timeout_ms == DEFAULT_TIMEOUT_MS else None
    entry = pool.checkout_entry() if pool is not None else None
    if entry is None:
        with fresh() as conn:
            yield conn
        return
    with entry.busy:
        try:
            if entry.closed:
                with fresh() as conn:
                    yield conn
                return
            tracked = _Tracked(entry.conn)
            try:
                yield tracked
            except BaseException:
                pool.forget_current()  # state unknown: never reused
                raise
            try:
                tracked._finish()
            except sqlite3.Error:
                pool.forget_current()
        finally:
            finish_entry(entry)  # closed while in use: its own thread closes it


def close_path(path: Path | str) -> None:
    """Close every pooled snapshot connection to one store file."""
    with _pools_lock:
        pool = _pools.pop(str(Path(path).expanduser().resolve()), None)
    if pool is not None:
        pool.close_all()


def close_all() -> None:
    """Close every pooled snapshot connection, waiting one bounded time in total."""
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    deadline = time.monotonic() + _rcp.CLOSE_WAIT_SECONDS
    for pool in pools:
        pool.close_all(deadline)


def _reset_after_fork() -> None:
    """In a forked child: the pools reset themselves; the registry lock may be held."""
    global _pools_lock
    _pools_lock = threading.Lock()
    _pools.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def open_count(path: Path | str) -> int:
    with _pools_lock:
        pool = _pools.get(str(Path(path).expanduser().resolve()))
    return pool.open_count() if pool is not None else 0


__all__ = ["DEFAULT_TIMEOUT_MS", "MAX_PATHS", "close_all", "close_path", "open_count", "snapshot"]
