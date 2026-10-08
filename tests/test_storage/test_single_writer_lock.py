# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""TDD — single-writer lock across all memory.db writers.

GATE BLOCKER
============
Before v3.8.4, DatabaseManager, VectorStore, adapter_base.sync_log_record,
and several learning modules each opened their own sqlite3.connect() to
memory.db WITHOUT holding a shared Python lock.  Under WAL mode only ONE
writer can hold the write lock at any instant; concurrent Python threads
competing for it produce SQLITE_BUSY errors that trigger
DatabaseManager._execute_one's 5x 10-second retry loop — up to 50-second
stalls on user /remember calls.

Fix: write_lock.get_write_lock(db_path) returns a per-path process-level
RLock.  Every in-process writer acquires this lock BEFORE opening sqlite3.
DatabaseManager._lock IS this same lock (via get_write_lock in __init__).

TDD Contract
============
RED  (tree 2f35ddc — no shared write lock):
  * test_db_manager_lock_identity          FAILS (new RLock per DM instance,
                                            not equal to get_write_lock result)
  * test_vector_store_waits_for_write_lock FAILS (vs.upsert() opens conn
                                            instantly, doesn't wait for lock)
                                            [skipped if sqlite-vec unavailable]
  * test_adapter_sync_waits_for_write_lock FAILS (sync_log_record opens conn
                                            without acquiring write lock)

GREEN (after fix):
  * All three PASS — shared lock object, and writers serialised.
  * test_realistic_concurrent_writes PASSES — zero "database is locked"
    errors under real concurrent load, every write lands, none is retried.
    It waits while the write lock keeps changing hands (measured turns), not
    for a fixed number of seconds: a fixed bound measured the disk.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.write_lock import get_write_lock
from tests.test_storage._write_lock_probe import (
    install_timed_lock,
    remove_timed_lock,
    wait_for_writers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_db(tmp_path: Path) -> Path:
    """Return path to a fresh WAL-mode memory.db with base schema."""
    db_path = tmp_path / "memory.db"
    db = DatabaseManager(str(db_path))
    db.initialize(real_schema)
    return db_path


def _count_rows(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def _minimal_schema_conn(db_path: Path) -> sqlite3.Connection:
    """Open a raw conn and ensure the atomic_facts table exists."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS atomic_facts ("
        "fact_id TEXT PRIMARY KEY, "
        "profile_id TEXT NOT NULL, "
        "content TEXT NOT NULL, "
        "created_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Test 1 — Lock identity
#
# RED:  database.py creates `self._lock = threading.RLock()` (new per instance)
#       → two separate DMs have different lock objects
#       → get_write_lock(db_path) returns yet another new lock → all 3 differ
# GREEN: database.py does `self._lock = get_write_lock(self.db_path)`
#       → same resolved path → same RLock object for both DMs and the
#       direct get_write_lock() call
# ---------------------------------------------------------------------------

def test_db_manager_lock_identity(tmp_path: pytest.TempPathFactory) -> None:
    """DatabaseManager._lock must be the SAME object as get_write_lock(db_path).

    Two DMs for the same path must share one RLock so that all in-process
    writes are serialised through that single Python lock.
    """
    db_path = tmp_path / "memory.db"

    db1 = DatabaseManager(str(db_path))
    db2 = DatabaseManager(str(db_path))
    registry_lock = get_write_lock(db_path)

    # All three must be the SAME Python object.
    assert db1._lock is registry_lock, (
        "db1._lock is not the process-level write lock — "
        "DatabaseManager is creating a new RLock() instead of calling "
        "get_write_lock().  This means db1 writes bypass the shared lock."
    )
    assert db2._lock is registry_lock, (
        "db2._lock is not the process-level write lock — "
        "same issue for a second DatabaseManager instance on the same path."
    )
    assert db1._lock is db2._lock, (
        "db1._lock and db2._lock are different objects — "
        "two threads using different DMs for the same DB will race."
    )


# ---------------------------------------------------------------------------
# Test 2 — VectorStore.upsert() must wait for a held write lock
#
# RED:  VectorStore.upsert() opens its own sqlite3.connect() WITHOUT calling
#       get_write_lock() — it does not wait for the held lock, so the elapsed
#       time will be ≈ 0 ms, much less than HOLD_SEC.
# GREEN: VectorStore.upsert() calls `with get_write_lock(db_path)` before
#        opening the connection — it blocks for ≥ HOLD_SEC.
#
# Skipped automatically if sqlite-vec is not installed (VectorStore falls
# back to ANNIndex; upsert() is a no-op when not available).
# ---------------------------------------------------------------------------

HOLD_SEC = 0.35  # how long the main thread holds the write lock


@pytest.mark.skipif(
    pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed") is None,
    reason="sqlite-vec not installed — VectorStore is a no-op",
)
def test_vector_store_waits_for_write_lock(tmp_path: Path) -> None:
    """VectorStore.upsert() must serialise with the process-level write lock.

    The main thread holds get_write_lock for HOLD_SEC.  A background thread
    calling vs.upsert() must NOT proceed (open its sqlite3 connection) until
    the lock is released.  If it does proceed immediately — as it did before
    the fix — elapsed < HOLD_SEC * 0.8 and the test fails.
    """
    pytest.importorskip("sqlite_vec")

    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    db_path = _fresh_db(tmp_path)
    config = VectorStoreConfig(dimension=4, enabled=True)
    vs = VectorStore(db_path, config)
    if not vs.available:
        pytest.skip("sqlite-vec extension not loadable on this machine")

    upsert_started_at: list[float] = []

    def _upsert() -> None:
        upsert_started_at.append(time.monotonic())
        vs.upsert("fact-001", "default", [0.1, 0.2, 0.3, 0.4])

    # Hold the write lock for HOLD_SEC so the upsert thread must wait.
    lock = get_write_lock(db_path)
    t = threading.Thread(target=_upsert, daemon=True)
    with lock:
        t.start()
        time.sleep(HOLD_SEC)

    t.join(timeout=5.0)

    assert upsert_started_at, "upsert thread never recorded start time"
    # The upsert thread logs its start time BEFORE acquiring the lock —
    # but the actual sqlite3 write only happens AFTER the lock is released.
    # We can't time the exact lock-acquire moment from outside, so instead
    # we verify the thread joined *after* the lock was released.
    # A tighter check: the thread should not FINISH before the lock is
    # released (it would finish nearly instantly if it bypassed the lock).
    elapsed = time.monotonic() - upsert_started_at[0]
    assert elapsed >= HOLD_SEC * 0.8, (
        f"VectorStore.upsert() completed in {elapsed:.3f}s, but the write "
        f"lock was held for {HOLD_SEC}s.  This means upsert() bypassed the "
        f"process-level write lock and opened a direct sqlite3 connection "
        f"without serialisation — the pre-fix race condition."
    )


# ---------------------------------------------------------------------------
# Test 3 — adapter_base.sync_log_record() must wait for a held write lock
#
# RED:  sync_log_record() opens sqlite3.connect() without get_write_lock()
#       → does not wait for the held lock → elapsed ≈ 0 ms → FAIL
# GREEN: sync_log_record() wraps its conn in `with get_write_lock(db_path):`
#        → blocks for ≥ HOLD_SEC → PASS
# ---------------------------------------------------------------------------

def test_adapter_sync_waits_for_write_lock(tmp_path: Path) -> None:
    """adapter_base.sync_log_record() must serialise with the write lock.

    Simulates the IDE-adapter sync thread writing cross_platform_sync_log
    rows while the main thread holds the write lock (e.g. a user /remember
    in progress).  Before the fix, sync_log_record opened its own
    sqlite3.connect() with no lock at all — it would write immediately and
    compete with DatabaseManager's WAL write lock, causing SQLITE_BUSY.
    """
    from superlocalmemory.hooks.adapter_base import path_sha256, sync_log_record

    db_path = tmp_path / "memory.db"

    sync_started_at: list[float] = []
    errors: list[Exception] = []

    def _sync() -> None:
        try:
            sync_started_at.append(time.monotonic())
            target_path = tmp_path / "cursor.md"
            sync_log_record(
                db_path,
                adapter_name="cursor_project",
                profile_id="default",
                target_path_sha256=path_sha256(target_path),
                target_basename="cursor.md",
                bytes_written=42,
                content_sha256="a" * 64,
                success=True,
            )
        except Exception as exc:
            errors.append(exc)

    lock = get_write_lock(db_path)
    t = threading.Thread(target=_sync, daemon=True)
    with lock:
        t.start()
        time.sleep(HOLD_SEC)
    t.join(timeout=5.0)

    assert not errors, f"sync_log_record raised: {errors}"
    assert sync_started_at, "sync thread never recorded start time"
    elapsed = time.monotonic() - sync_started_at[0]
    assert elapsed >= HOLD_SEC * 0.8, (
        f"sync_log_record() completed in {elapsed:.3f}s, but the write lock "
        f"was held for {HOLD_SEC}s.  This means sync_log_record() opened a "
        f"direct sqlite3 connection without acquiring the process-level write "
        f"lock — the pre-fix race condition that caused SQLITE_BUSY stalls."
    )


# ---------------------------------------------------------------------------
# Test 4 — Realistic concurrent writes: zero "database is locked" errors
#
# Simulates the actual race condition:
#   - N_ADAPTER_THREADS adapter-sync threads writing cross_platform_sync_log
#   - 1 user thread performing db.execute() writes (atomic_facts INSERTs)
# Under the fix, all writers share ONE Python lock → no SQLite WAL contention
# → no "database is locked" errors → user store completes in < 2 s.
#
# RED:  Without the write lock, adapter threads and the DM thread compete at
#       the WAL level → SQLITE_BUSY → db.execute() hangs in the retry loop
#       → user store takes > 10 s (or raises).
# GREEN: All writers serialised through the Python lock → zero errors, < 2 s.
# ---------------------------------------------------------------------------

N_ADAPTER_THREADS = 4
N_ADAPTER_OPS_PER_THREAD = 30
N_USER_OPS = 20


def test_realistic_concurrent_writes(tmp_path: Path, caplog, request) -> None:
    """Zero 'database is locked' errors under realistic concurrent write load.

    Asserts:
      1. No thread encounters a SQLITE_BUSY / 'database is locked' error.
      2. All N_USER_OPS user-store writes complete.
      3. DatabaseManager's busy-retry loop never fires: the writers are
         serialised, not retried. This used to be inferred from the run
         taking under 2 s, which measured the disk instead: a Windows runner
         commits slowly enough to take 2.7 s with no contention at all.
    """
    import logging

    caplog.set_level(logging.DEBUG, logger="superlocalmemory.storage.database")
    from superlocalmemory.hooks.adapter_base import path_sha256, sync_log_record

    # Every writer of this path gets the timed lock (installed before the
    # first DatabaseManager), so the wait below follows measured turns.
    probe = install_timed_lock(tmp_path / "memory.db")
    request.addfinalizer(lambda: remove_timed_lock(tmp_path / "memory.db"))
    db_path = _fresh_db(tmp_path)

    # Seed required FK parents: profile + memory.
    # We reuse one memory_id for all atomic_facts INSERTs in the test.
    setup_db = DatabaseManager(str(db_path))
    setup_db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "default"),
    )
    test_mem_id = str(uuid.uuid4())
    setup_db.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES (?, ?, ?)",
        (test_mem_id, "default", "seed memory for concurrent test"),
    )

    locked_errors: list[str] = []
    adapter_errors: list[Exception] = []
    user_store_errors: list[Exception] = []
    user_store_times: list[float] = []

    stop_event = threading.Event()

    # -- Adapter-sync threads ------------------------------------------------
    def _adapter_worker(thread_idx: int) -> None:
        target_path = tmp_path / f"adapter_{thread_idx}.md"
        tsha = path_sha256(target_path)
        for op_idx in range(N_ADAPTER_OPS_PER_THREAD):
            if stop_event.is_set():
                break
            try:
                sync_log_record(
                    db_path,
                    adapter_name=f"cursor_project_{thread_idx}",
                    profile_id="default",
                    target_path_sha256=tsha,
                    target_basename=f"adapter_{thread_idx}.md",
                    bytes_written=op_idx * 10,
                    content_sha256=f"{op_idx:064x}",
                    success=True,
                )
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" in msg or "busy" in msg:
                    locked_errors.append(
                        f"adapter[{thread_idx}] op {op_idx}: {exc}"
                    )
                else:
                    adapter_errors.append(exc)
            except Exception as exc:
                adapter_errors.append(exc)
            time.sleep(0.001)  # realistic ~1 ms cadence

    # -- User-store thread ---------------------------------------------------
    def _user_store_worker() -> None:
        db = DatabaseManager(str(db_path))
        for i in range(N_USER_OPS):
            fact_id = str(uuid.uuid4())
            t0 = time.monotonic()
            try:
                db.execute(
                    "INSERT INTO atomic_facts "
                    "(fact_id, memory_id, profile_id, content) VALUES (?, ?, ?, ?)",
                    (fact_id, test_mem_id, "default", f"test fact {i}"),
                )
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" in msg or "busy" in msg:
                    locked_errors.append(f"user_store op {i}: {exc}")
                else:
                    user_store_errors.append(exc)
            except Exception as exc:
                user_store_errors.append(exc)
            user_store_times.append(time.monotonic() - t0)
            time.sleep(0.005)  # 5 ms between user ops

    # Launch adapter threads first, then user thread.
    adapter_threads = [
        threading.Thread(target=_adapter_worker, args=(i,), daemon=True)
        for i in range(N_ADAPTER_THREADS)
    ]
    for t in adapter_threads:
        t.start()

    user_thread = threading.Thread(
        target=_user_store_worker, name="user-store", daemon=True,
    )
    t_start = time.monotonic()
    user_thread.start()
    # Not a fixed wall-clock join: on a loaded machine 140 serialised commits
    # can take longer than any fixed bound while every writer is still taking
    # its turn. Wait as long as the lock keeps changing hands; fail on a stall.
    try:
        wait_for_writers([user_thread], probe)
    finally:
        stop_event.set()
    wait_for_writers(adapter_threads, probe)

    total_user_time = time.monotonic() - t_start

    # Assertions
    assert not locked_errors, (
        f"Got {len(locked_errors)} 'database is locked' errors:\n"
        + "\n".join(locked_errors[:5])
    )
    assert not user_store_errors, (
        f"User-store raised unexpected errors: {user_store_errors[:3]}"
    )
    assert not adapter_errors, (
        f"Adapter threads raised unexpected errors: {adapter_errors[:3]}"
    )
    # All user ops must have completed (wait_for_writers already failed on a
    # stall, so a shortfall here means a write was dropped, not still queued).
    assert len(user_store_times) == N_USER_OPS, (
        f"Only {len(user_store_times)}/{N_USER_OPS} user-store ops completed."
    )
    assert _count_rows(db_path, "atomic_facts") == N_USER_OPS
    # Every adapter wrote too (one upserted row each; they stop with the user).
    assert _count_rows(db_path, "cross_platform_sync_log") == N_ADAPTER_THREADS
    # The retry loop logs each busy retry before it backs off (0.1 s and up).
    retries = [r.getMessage() for r in caplog.records if "DB busy" in r.getMessage()]
    assert not retries, (
        f"DatabaseManager retried {len(retries)} busy writes in "
        f"{total_user_time:.2f}s: the adapter threads and the user-store thread "
        f"still contend. First: {retries[0]}"
    )
