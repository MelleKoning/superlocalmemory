# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""A save that meets busy writers waits its turn; it is never refused.

The slowness is injected, not hoped for: competing writers hold the shared
memory.db write lock for a fixed time per turn, which is what a loaded disk
does to every commit. And the adapter sync, the background writer the user's
save queues behind, keeps its turn short: one connection, one commit, and no
checkpoint or WAL deletion when it closes.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from superlocalmemory.hooks import adapter_base
from superlocalmemory.hooks.adapter_base import path_sha256, sync_log_record
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.write_lock import get_write_lock
from tests.test_storage._write_lock_probe import (
    install_timed_lock,
    remove_timed_lock,
    wait_for_writers,
)

SLOW_TURN_S = 0.03
N_SLOW_WRITERS = 4
N_SLOW_TURNS = 10
N_USER_SAVES = 10


def _record(db_path: Path, adapter: str, op: int) -> None:
    sync_log_record(
        db_path,
        adapter_name=adapter,
        profile_id="default",
        target_path_sha256=path_sha256(db_path.parent / f"{adapter}.md"),
        target_basename=f"{adapter}.md",
        bytes_written=op,
        content_sha256=f"{op:064x}",
        success=True,
    )


@pytest.fixture()
def seeded(tmp_path: Path, request: pytest.FixtureRequest):
    db_path = tmp_path / "memory.db"
    probe = install_timed_lock(db_path)
    request.addfinalizer(lambda: remove_timed_lock(db_path))
    db = DatabaseManager(str(db_path))
    db.initialize(real_schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default', 'default')")
    memory_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES (?, 'default', 'seed')",
        (memory_id,),
    )
    db.close()
    return db_path, memory_id, probe


def test_sync_log_record_opens_one_connection_per_record(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    DatabaseManager(str(db_path)).initialize(real_schema)
    _record(db_path, "warmup", 0)  # the table exists from here on

    opened: list[str] = []
    real_connect = sqlite3.connect

    def counting_connect(*args, **kwargs):
        opened.append(str(args[0]) if args else "")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(adapter_base.sqlite3, "connect", counting_connect)
    _record(db_path, "cursor", 1)
    assert len(opened) == 1, (
        f"one sync-log record opened {len(opened)} connections inside the "
        "write lock; every extra open/close lengthens the turn a save waits for"
    )


def test_sync_log_record_does_not_checkpoint_on_close(tmp_path):
    """Closing the adapter's connection must not checkpoint and delete the WAL.

    With no other connection open, a default close checkpoints the WAL into
    the database, fsyncs it and deletes the WAL and its index - all inside the
    write lock. The WAL surviving the close is the observable proof it did not.
    """
    db_path = tmp_path / "memory.db"
    DatabaseManager(str(db_path)).initialize(real_schema)
    wal = Path(str(db_path) + "-wal")

    _record(db_path, "cursor", 1)

    assert wal.exists() and wal.stat().st_size > 0, (
        "the adapter's close checkpointed and deleted the WAL while holding "
        "the shared write lock"
    )
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT bytes_written FROM cross_platform_sync_log").fetchall()
    finally:
        conn.close()
    assert rows == [(1,)]


def test_a_save_behind_slow_writers_waits_its_turn_and_lands(seeded):
    """Slow turns make a save wait; they never make it fail or jump the queue.

    Each competing writer holds the lock for SLOW_TURN_S per turn, so the run
    takes about N_SLOW_WRITERS x N_SLOW_TURNS x SLOW_TURN_S of held lock. Every
    user save must land, none may hit "database is locked", and none may be
    overtaken by more turns than there are competing writers plus one (the
    turn already in progress when it asked): it waits its turn, it is not
    starved.
    """
    db_path, memory_id, probe = seeded
    errors: list[BaseException] = []
    overtaken: list[int] = []
    stop = threading.Event()

    def slow_writer(index: int) -> None:
        for op in range(N_SLOW_TURNS):
            if stop.is_set():
                return
            try:
                with get_write_lock(db_path):
                    time.sleep(SLOW_TURN_S)  # a slow disk, held inside the turn
                    _record(db_path, f"slow_{index}", op)
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)
            time.sleep(0.001)

    def user_saves() -> None:
        db = DatabaseManager(str(db_path))
        try:
            for i in range(N_USER_SAVES):
                before = len(probe.holds)
                try:
                    db.execute(
                        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                        "VALUES (?, ?, 'default', ?)",
                        (str(uuid.uuid4()), memory_id, f"user fact {i}"),
                    )
                except BaseException as exc:  # noqa: BLE001 - asserted below
                    errors.append(exc)
                # Turns that ended between asking and finishing, minus our own.
                overtaken.append(len(probe.holds) - before - 1)
                time.sleep(0.005)
        finally:
            db.close()

    writers = [
        threading.Thread(target=slow_writer, args=(i,), name=f"slow-{i}", daemon=True)
        for i in range(N_SLOW_WRITERS)
    ]
    for thread in writers:
        thread.start()
    user = threading.Thread(target=user_saves, name="user-saves", daemon=True)
    user.start()
    try:
        wait_for_writers([user], probe)
    finally:
        stop.set()
    wait_for_writers(writers, probe)

    assert errors == []
    conn = sqlite3.connect(str(db_path))
    try:
        saved = conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0]
    finally:
        conn.close()
    assert saved == N_USER_SAVES
    assert max(overtaken) <= N_SLOW_WRITERS + 1, (
        f"a save was overtaken by {max(overtaken)} turns with only "
        f"{N_SLOW_WRITERS} competing writers: it was starved, not queued "
        f"(per save: {overtaken})"
    )
