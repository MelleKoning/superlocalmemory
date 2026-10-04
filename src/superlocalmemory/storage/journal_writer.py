# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Group-commit writer and pooled readers for the admission journal.

Every journal mutation used to open its own SQLite connection, take a process
lock, and pay its own FULL-synchronous commit. Under a burst of concurrent
saves the callers queued on that lock until their budget ran out and the save
was refused with nothing stored.

Here one thread (``slm-journal-writer``) owns one persistent WAL connection.
Callers hand it small operations; it gathers up to ``max_batch`` of them
(lingering a couple of milliseconds for company), runs each inside its own
SAVEPOINT so one failure cannot spoil the rest, and makes the whole batch
durable with a single COMMIT. A batch is atomic: a crash leaves all of it or
none of it.

Cancel before commit. A caller whose deadline passes before its operation is
committed withdraws it, and the writer guarantees a withdrawn operation is
never committed: if a withdrawal lands while its batch is executing, the batch
is rolled back and re-run without it. Once the writer has claimed the batch
for COMMIT, withdrawal is no longer possible and the caller waits for the real
outcome, so a caller is never told "not saved" about something that was.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger("superlocalmemory.storage.journal_writer")

MAX_BATCH = 64
LINGER_SECONDS = 0.002
QUEUE_CAP = 1024
READ_POOL_SIZE = 4
#: SQLite wait for an operation that has no caller deadline (replay, tests).
_UNBOUNDED_BUSY_SECONDS = 5.0
#: An idle writer thread exits and closes its connection; the next operation
#: starts it again. Keeps an idle daemon (and every test) free of a parked thread.
_IDLE_EXIT_SECONDS = 5.0
#: Seconds a caller refused by a full queue is told to wait before retrying.
OVERLOAD_RETRY_AFTER_SECONDS = 1


class AdmissionJournalUnavailable(RuntimeError):
    """The durable admission journal could not mutate within its caller budget.

    Raised only when the operation is guaranteed NOT to have been committed.
    """


class AdmissionJournalOverloaded(AdmissionJournalUnavailable):
    """More saves are queued for the journal than it accepts at once."""

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


_BUSY_MESSAGE = "admission journal is busy beyond its caller deadline"

Operation = Callable[[sqlite3.Connection], Any]


@dataclass(eq=False)
class _Op:
    fn: Operation
    deadline: float | None
    done: threading.Event = field(default_factory=threading.Event)
    # queued -> running -> committing -> finished, or -> cancelled
    state: str = "queued"
    result: Any = None
    error: BaseException | None = None

    def expired(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        self.result, self.error = result, error
        self.state = "finished"
        self.done.set()


def is_sqlite_busy(error: sqlite3.Error) -> bool:
    """Recognize primary and extended SQLite BUSY/LOCKED result codes."""
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(error).casefold()
    return "locked" in message or "busy" in message


def _connect(path: Path, *, timeout: float) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(path), timeout=max(0.001, timeout), isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class GroupCommitWriter:
    """One thread, one connection, one COMMIT per batch of journal mutations."""

    def __init__(
        self,
        path: Path,
        *,
        max_batch: int = MAX_BATCH,
        linger_seconds: float = LINGER_SECONDS,
        queue_cap: int = QUEUE_CAP,
        name: str = "slm-journal-writer",
    ) -> None:
        self._path = Path(path)
        self._max_batch = max_batch
        self._linger = linger_seconds
        self._queue_cap = queue_cap
        self._name = name
        self._cond = threading.Condition()
        self._queue: deque[_Op] = deque()
        self._thread: threading.Thread | None = None
        self._closing = False
        #: Test seams: called with the live batch immediately before COMMIT
        #: (after it was claimed) and immediately after COMMIT (before any
        #: caller is told). Crash tests kill the process from them.
        self.before_commit: Callable[[list[_Op]], None] | None = None
        self.after_commit: Callable[[list[_Op]], None] | None = None

    # -- caller side -----------------------------------------------------
    def submit(self, fn: Operation, *, deadline: float | None = None) -> Any:
        """Run ``fn(conn)`` in the next batch; return its result once durable."""
        op = _Op(fn=fn, deadline=deadline)
        with self._cond:
            if len(self._queue) >= self._queue_cap:
                raise AdmissionJournalOverloaded(
                    "too many saves are waiting for the admission journal",
                    retry_after_seconds=OVERLOAD_RETRY_AFTER_SECONDS,
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise AdmissionJournalUnavailable(_BUSY_MESSAGE)
            self._queue.append(op)
            self._ensure_thread_locked()
            self._cond.notify_all()
        self._wait(op)
        if op.error is not None:
            raise op.error
        return op.result

    def _wait(self, op: _Op) -> None:
        remaining = None if op.deadline is None else max(0.0, op.deadline - time.monotonic())
        if op.done.wait(remaining):
            return
        with self._cond:
            if op.state in {"queued", "running"}:
                op.state = "cancelled"
                op.error = AdmissionJournalUnavailable(_BUSY_MESSAGE)
                return
        # Claimed for COMMIT: the outcome is decided by the disk, not the clock.
        op.done.wait()

    def close(self, timeout: float = 5.0) -> None:
        """Drain queued operations, stop the thread, close the connection."""
        with self._cond:
            self._closing = True
            thread = self._thread
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._cond:
            self._closing = False
            if self._queue:  # submitted during close: serve them
                self._ensure_thread_locked()

    def _ensure_thread_locked(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    # -- writer side -----------------------------------------------------
    def _run(self) -> None:
        conn: sqlite3.Connection | None = None
        try:
            while True:
                batch = self._take_batch()
                if batch is None:
                    return
                try:
                    if conn is None:
                        conn = _connect(self._path, timeout=_UNBOUNDED_BUSY_SECONDS)
                        conn.execute("PRAGMA synchronous=FULL")
                    self._run_batch(conn, batch)
                except BaseException as exc:  # noqa: BLE001 - never strand a caller
                    logger.error("admission journal batch failed (%s)", type(exc).__name__)
                    _fail(batch, exc)
                    if conn is not None:
                        _safe_rollback(conn)
                        conn.close()
                        conn = None
        finally:
            if conn is not None:
                conn.close()

    def _take_batch(self) -> list[_Op] | None:
        with self._cond:
            idle_until = time.monotonic() + _IDLE_EXIT_SECONDS
            while not self._queue:
                if self._closing:
                    return None
                remaining = idle_until - time.monotonic()
                if remaining <= 0:
                    self._thread = None  # a later submit starts a fresh one
                    return None
                self._cond.wait(remaining)
            # Linger briefly for company: concurrent savers arriving now share
            # this batch's single COMMIT instead of paying their own. Measured
            # on perf_counter so a frozen monotonic clock cannot stall it.
            linger_until = time.perf_counter() + self._linger
            while len(self._queue) < self._max_batch and not self._closing:
                remaining = linger_until - time.perf_counter()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            batch: list[_Op] = []
            while self._queue and len(batch) < self._max_batch:
                batch.append(self._queue.popleft())
            return batch

    def _claim(self, ops: list[_Op]) -> list[_Op]:
        """Drop withdrawn/expired operations; mark the rest running."""
        now = time.monotonic()
        live: list[_Op] = []
        with self._cond:
            for op in ops:
                if op.state == "cancelled":
                    op.done.set()
                elif op.expired(now):
                    op.state = "cancelled"
                    op.finish(error=AdmissionJournalUnavailable(_BUSY_MESSAGE))
                else:
                    op.state = "running"
                    live.append(op)
        return live

    def _run_batch(self, conn: sqlite3.Connection, batch: list[_Op]) -> None:
        live = self._claim(batch)
        while live:
            if not self._begin(conn, live):
                live = self._claim(live)
                continue
            outcomes = self._execute(conn, live)
            if outcomes is None:  # SQLite aborted the transaction itself
                return
            with self._cond:
                withdrawn = [op for op in live if op.state == "cancelled"]
                if not withdrawn:
                    for op in live:
                        op.state = "committing"
            if withdrawn:
                # Cancel before commit: undo the whole batch and redo it
                # without the withdrawn operations.
                _safe_rollback(conn)
                live = self._claim(live)
                continue
            if self.before_commit is not None:
                self.before_commit(live)
            try:
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                _safe_rollback(conn)
                _fail(live, exc)
                return
            if self.after_commit is not None:
                self.after_commit(live)
            for op, (result, error) in zip(live, outcomes):
                op.finish(result, error)
            return

    def _begin(self, conn: sqlite3.Connection, live: list[_Op]) -> bool:
        deadlines = [op.deadline for op in live]
        if any(d is None for d in deadlines):
            wait = _UNBOUNDED_BUSY_SECONDS
        else:
            wait = max(0.001, max(deadlines) - time.monotonic())  # type: ignore[type-var]
        conn.execute(f"PRAGMA busy_timeout={max(1, int(wait * 1000))}")
        try:
            conn.execute("BEGIN IMMEDIATE")
            return True
        except sqlite3.OperationalError as exc:
            if not is_sqlite_busy(exc):
                raise
        # Another process holds the journal: fail what is now out of time,
        # retry the rest (callers withdraw themselves at their deadline).
        now = time.monotonic()
        with self._cond:
            for op in live:
                if op.state == "running" and (op.deadline is None or op.expired(now)):
                    op.state = "cancelled"
                    op.finish(error=AdmissionJournalUnavailable(_BUSY_MESSAGE))
        return False

    def _execute(
        self, conn: sqlite3.Connection, live: list[_Op],
    ) -> list[tuple[Any, BaseException | None]] | None:
        outcomes: list[tuple[Any, BaseException | None]] = []
        for index, op in enumerate(live):
            savepoint = f"journal_op_{index}"
            conn.execute(f"SAVEPOINT {savepoint}")
            try:
                outcomes.append((op.fn(conn), None))
                conn.execute(f"RELEASE {savepoint}")
            except BaseException as exc:  # noqa: BLE001 - reported to its caller
                if not conn.in_transaction:
                    # SQLite rolled the whole transaction back (I/O error, full
                    # disk). Nothing in this batch is saved; say so to all.
                    _fail(live, exc)
                    return None
                conn.execute(f"ROLLBACK TO {savepoint}")
                conn.execute(f"RELEASE {savepoint}")
                outcomes.append((None, exc))
        return outcomes


def _fail(ops: list[_Op], cause: BaseException) -> None:
    for op in ops:
        if op.done.is_set() and op.state == "finished":
            continue
        if isinstance(cause, AdmissionJournalUnavailable):
            error: BaseException = cause
        else:
            error = AdmissionJournalUnavailable("admission journal could not commit")
            error.__cause__ = cause
        op.finish(error=error)


def _safe_rollback(conn: sqlite3.Connection) -> None:
    try:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
    except sqlite3.Error:
        pass


class ReadPool:
    """A bounded set of persistent read connections to the journal."""

    def __init__(self, path: Path, *, size: int = READ_POOL_SIZE) -> None:
        self._path = Path(path)
        self._size = size
        self._idle: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue()
        self._lock = threading.Lock()
        self._created = 0

    @contextmanager
    def connection(
        self, *, deadline: float | None = None,
    ) -> Generator[sqlite3.Connection, None, None]:
        conn = self._acquire(deadline)
        healthy = True
        try:
            remaining = 1.0 if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise AdmissionJournalUnavailable(_BUSY_MESSAGE)
            conn.execute(f"PRAGMA busy_timeout={max(1, int(remaining * 1000))}")
            yield conn
        except sqlite3.OperationalError as exc:
            if is_sqlite_busy(exc):
                raise AdmissionJournalUnavailable(_BUSY_MESSAGE) from exc
            healthy = False
            raise
        except sqlite3.DatabaseError:
            healthy = False
            raise
        finally:
            if conn.in_transaction:
                _safe_rollback(conn)
            self._release(conn, healthy)

    def _acquire(self, deadline: float | None) -> sqlite3.Connection:
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self._size:
                self._created += 1
                try:
                    return _connect(self._path, timeout=1.0)
                except BaseException:
                    self._created -= 1
                    raise
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            return self._idle.get(timeout=timeout)
        except queue.Empty as exc:
            raise AdmissionJournalUnavailable(_BUSY_MESSAGE) from exc

    def _release(self, conn: sqlite3.Connection, healthy: bool) -> None:
        if healthy:
            self._idle.put(conn)
            return
        conn.close()
        with self._lock:
            self._created -= 1

    def close(self) -> None:
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                return
            conn.close()
            with self._lock:
                self._created -= 1
