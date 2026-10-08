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
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

#: Connections held at once, one per reading thread (~3 MB each: schema + cache).
MAX_THREADS = 32
#: Words that can begin the statement a ``WITH`` clause introduces.
_MAIN_WORDS = frozenset({"SELECT", "VALUES", "INSERT", "UPDATE", "DELETE", "REPLACE"})
_READ_WORDS = frozenset({"SELECT", "VALUES"})


def with_main_word(sql: str) -> str:
    """The first word of the statement after a ``WITH`` clause; "" if unknown.

    Common table expressions sit in parentheses, so the main statement is the
    first SELECT / VALUES / INSERT / UPDATE / DELETE / REPLACE outside them.
    Quoted text and comments are skipped.
    """
    depth, i, n = 0, 0, len(sql)
    while i < n:
        c = sql[i]
        if c in "'\"`[":
            close = "]" if c == "[" else c
            j = sql.find(close, i + 1)
            while j != -1 and close != "]" and sql.startswith(close, j + 1):
                j = sql.find(close, j + 2)  # a doubled quote is an escaped quote
            if j == -1:
                return ""
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j + 1
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            if depth == 0 and sql[i:j].upper() in _MAIN_WORDS:
                return sql[i:j].upper()
            i = j
        else:
            depth += (c == "(") - (c == ")")
            i += 1
    return ""


def is_plain_read(sql: str) -> bool:
    """True for statements that may run on a reused, query-only connection.

    ``WITH ... DELETE`` (or any other write after a ``WITH``) is not a read: it
    must take the store's single-writer path.
    """
    head = sql.lstrip()
    word = head[:6].upper()
    if word.startswith("SELECT"):
        return True
    if word.startswith("WITH") and (len(head) == 4 or not (head[4].isalnum() or head[4] == "_")):
        return with_main_word(head[4:]) in _READ_WORDS
    return False


def is_write_with(sql: str) -> bool:
    """True for a ``WITH`` statement whose main statement is not a plain read."""
    head = sql.lstrip()
    return head[:4].upper() == "WITH" and not is_plain_read(head)


@dataclass
class _Entry:
    conn: sqlite3.Connection
    file_id: tuple[int, int] | None
    thread: threading.Thread
    closed: bool = field(default=False)
    #: Held while a statement runs, so ``close_all`` on another thread waits for it.
    busy: threading.Lock = field(default_factory=threading.Lock)
    #: The process that opened it: a forked child never uses its parent's connection.
    pid: int = field(default_factory=os.getpid)

#: Longest ``close_all`` waits, in total, for statements in flight. A connection
#: still busy at the deadline is closed by its own thread when the statement ends.
CLOSE_WAIT_SECONDS = 10.0

#: Every pool of this process, so a forked child can drop what it inherited.
_POOLS: "weakref.WeakSet[ReadConnectionPool]" = weakref.WeakSet()
#: Connections a forked child inherited. Kept referenced and never used or
#: closed there: SQLite connections must not cross a fork, and closing one in
#: the child could disturb the parent's locks on the same file.
_INHERITED: list[_Entry] = []


def finish_entry(entry: _Entry) -> None:
    """Called by the entry's own thread, holding ``busy``, when a statement ends.

    ``close_all`` may have given up waiting while the statement ran; the
    connection is closed here instead of being left open until its thread dies.
    """
    if entry.closed:
        try:
            entry.conn.close()
        except sqlite3.Error:
            pass


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
        _POOLS.add(self)

    def _reset_after_fork(self) -> None:
        """In a forked child: forget every inherited connection without touching it."""
        _INHERITED.extend(self._entries.values())
        self._lock = threading.Lock()  # may have been held by a parent thread
        self._entries = {}
        self._local = threading.local()

    # -- checkout ----------------------------------------------------------

    def checkout(self) -> sqlite3.Connection | None:
        """This thread's connection, or None when the caller must open its own."""
        entry = self.checkout_entry()
        return entry.conn if entry is not None else None

    def checkout_entry(self) -> _Entry | None:
        file_id = self._file_id()
        entry: _Entry | None = getattr(self._local, "entry", None)
        if entry is not None and entry.pid != os.getpid():
            # Inherited across a fork without the reset (no at-fork hook here).
            _INHERITED.append(entry)
            self._local.entry = entry = None
        if entry is not None and not entry.closed and entry.file_id == file_id:
            return entry
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
        return entry

    def discard_current(self) -> None:
        """Close and forget this thread's connection (after an error)."""
        entry: _Entry | None = getattr(self._local, "entry", None)
        self._local.entry = None
        if entry is not None:
            with self._lock:
                self._entries.pop(id(entry), None)
            self._close(entry)

    def forget_current(self) -> None:
        """Drop this thread's connection while holding its busy lock (close it inline)."""
        entry: _Entry | None = getattr(self._local, "entry", None)
        self._local.entry = None
        if entry is not None:
            with self._lock:
                self._entries.pop(id(entry), None)
            entry.closed = True
            try:
                entry.conn.close()
            except sqlite3.Error:
                pass

    def close_all(self, deadline: float | None = None) -> None:
        """Close every thread's connection. The pool stays usable afterwards.

        Waits at most until ``deadline`` (``time.monotonic()``; default
        ``CLOSE_WAIT_SECONDS`` from now) in total, not per connection.
        """
        if deadline is None:
            deadline = time.monotonic() + CLOSE_WAIT_SECONDS
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            entry.closed = True  # every owner sees it before anyone waits
        for entry in entries:
            self._close(entry, deadline)

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
            # A dead thread holds no statement; never wait under the pool lock.
            self._close(self._entries.pop(key), time.monotonic())

    @staticmethod
    def _close(entry: _Entry, deadline: float | None = None) -> None:
        entry.closed = True  # its thread opens a fresh one from now on
        if deadline is None:
            deadline = time.monotonic() + CLOSE_WAIT_SECONDS
        if not entry.busy.acquire(timeout=max(0.0, deadline - time.monotonic())):
            return  # a statement is still running: its thread closes it (finish_entry)
        try:
            entry.conn.close()
        except sqlite3.Error:
            pass
        finally:
            entry.busy.release()


def _reset_all_after_fork() -> None:
    for pool in list(_POOLS):
        pool._reset_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_all_after_fork)


__all__ = ["MAX_THREADS", "ReadConnectionPool", "finish_entry", "is_plain_read",
           "is_write_with", "with_main_word"]


def execute_read(pool: ReadConnectionPool, sql: str, params: tuple,
                 fresh: Callable[[str, tuple], list], *, retries: int, base_delay: float) -> list:
    """Run a plain read on this thread's connection; ``fresh`` when none is available.

    Same contract as ``DatabaseManager._execute_one``: rows fully fetched, a
    transaction the statement opened is committed, "busy" retried with the same
    backoff. Any other error drops the connection before it propagates, so a
    connection in an unknown state is never used again.
    """
    entry = pool.checkout_entry()
    if entry is None:
        return fresh(sql, params)
    with entry.busy:
        if entry.closed:  # closed by another thread since the checkout
            finish_entry(entry)
            return fresh(sql, params)
        try:
            return _run(pool, entry.conn, sql, params, retries, base_delay)
        finally:
            finish_entry(entry)


def _run(pool: ReadConnectionPool, conn: sqlite3.Connection, sql: str, params: tuple,
         retries: int, base_delay: float) -> list:
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
            pool.forget_current()
            raise
        except BaseException:
            pool.forget_current()
            raise
    pool.forget_current()
    raise last  # type: ignore[misc]


__all__ += ["execute_read"]
