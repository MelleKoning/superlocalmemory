# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One reusable read connection per thread, for ``DatabaseManager.execute``.

WHY
---
``execute`` opened a new SQLite connection for every statement. Opening one
parses the whole schema (405 objects on a current store) and builds a page
cache, which is thousands of allocations -- and the SQLite this ships against
takes a process-wide lock for every allocation and for every page-cache
operation. Measured on a copy of a 22k-fact store, per statement: 4.26 ms with
one thread and 23.7 ms with six (a fresh connection each time), against
0.04 ms and 0.15 ms on a reused one. A recall runs dozens of statements on six
channel threads beside the daemon's background threads; native samples of the
slow recalls after a start showed most of their time waiting on exactly those
two locks, with unrelated small reads stalling together for 2.8-5 s.

WHAT THIS DOES
--------------
A plain read (``SELECT`` / ``WITH``) outside a transaction reuses its thread's
connection. Everything else -- writes, PRAGMAs, anything inside
``transaction()`` or ``raw_connection()`` -- keeps the per-statement connection
exactly as before. Answers cannot change: every statement runs in autocommit
and is fully consumed, so each one starts a fresh read snapshot and sees every
commit made before it, from any connection or process.

* A store file replaced underneath (a restore swaps ``memory.db``) is noticed by
  its device/inode before the next use: the old connection is closed, a new one
  opened.
* A connection that raised anything other than "busy" is dropped.
* At most ``MAX_THREADS`` threads hold one; beyond that a read falls back to a
  fresh connection. Connections of threads that have ended are closed on the
  next checkout.
* ``close_all`` (``DatabaseManager.close``) closes every thread's connection;
  garbage collection of the manager closes them too.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

#: Connections held at once, one per reading thread (~3 MB each: schema + cache).
MAX_THREADS = 32
_READ_PREFIXES = ("SELECT", "WITH")


def is_plain_read(sql: str) -> bool:
    """True for statements that may run on a reused connection."""
    head = sql.lstrip()[:6].upper()
    return head.startswith(_READ_PREFIXES)


@dataclass
class _Entry:
    conn: sqlite3.Connection
    file_id: tuple[int, int] | None
    thread: threading.Thread
    closed: bool = field(default=False)


class ReadConnectionPool:
    """Per-thread read connections for one database file."""

    def __init__(self, path: os.PathLike | str, opener: Callable[[], sqlite3.Connection],
                 *, max_threads: int = MAX_THREADS) -> None:
        self._path = os.fspath(path)
        self._open = opener
        self._max = max_threads
        self._local = threading.local()
        self._lock = threading.Lock()
        self._entries: dict[int, _Entry] = {}

    # -- checkout ----------------------------------------------------------

    def checkout(self) -> sqlite3.Connection | None:
        """This thread's connection, or None when the caller must open its own."""
        file_id = self._file_id()
        entry: _Entry | None = getattr(self._local, "entry", None)
        if entry is not None and not entry.closed and entry.file_id == file_id:
            return entry.conn
        if entry is not None:
            self.discard_current()
        if file_id is None:
            return None
        with self._lock:
            self._prune_dead_locked()
            if len(self._entries) >= self._max:
                return None
        conn = self._open()
        entry = _Entry(conn=conn, file_id=file_id, thread=threading.current_thread())
        with self._lock:
            self._entries[id(entry)] = entry
        self._local.entry = entry
        return conn

    def discard_current(self) -> None:
        """Close and forget this thread's connection (after an error)."""
        entry: _Entry | None = getattr(self._local, "entry", None)
        self._local.entry = None
        if entry is not None:
            with self._lock:
                self._entries.pop(id(entry), None)
            self._close(entry)

    def close_all(self) -> None:
        """Close every thread's connection. The pool stays usable afterwards."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._close(entry)

    def open_count(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- helpers -----------------------------------------------------------

    def _file_id(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self._path)
        except OSError:
            return None
        return (st.st_dev, st.st_ino)

    def _prune_dead_locked(self) -> None:
        dead = [k for k, e in self._entries.items() if not e.thread.is_alive()]
        for key in dead:
            self._close(self._entries.pop(key))

    @staticmethod
    def _close(entry: _Entry) -> None:
        entry.closed = True
        try:
            entry.conn.close()
        except sqlite3.Error:
            pass


__all__ = ["MAX_THREADS", "ReadConnectionPool", "is_plain_read"]


def execute_read(pool: ReadConnectionPool, sql: str, params: tuple,
                 fresh: Callable[[str, tuple], list], *, retries: int, base_delay: float) -> list:
    """Run a plain read on this thread's connection; ``fresh`` when none is available.

    Same contract as ``DatabaseManager._execute_one``: rows fully fetched, a
    transaction the statement opened is committed, "busy" retried with the same
    backoff. Any other error drops the connection before it propagates, so a
    connection in an unknown state is never used again.
    """
    import time

    conn = pool.checkout()
    if conn is None:
        return fresh(sql, params)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            rows = conn.execute(sql, params).fetchall()
            if conn.in_transaction:
                conn.commit()
            return rows
        except sqlite3.OperationalError as exc:
            text = str(exc).lower()
            if "locked" in text or "busy" in text:
                last = exc
                time.sleep(base_delay * (2 ** attempt))
                continue
            pool.discard_current()
            raise
        except BaseException:
            pool.discard_current()
            raise
    pool.discard_current()
    raise last  # type: ignore[misc]


__all__ += ["execute_read"]
