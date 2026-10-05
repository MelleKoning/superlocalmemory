# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The canonical writer's wait for another writer never outlives stop() or its caller.

The writer waited for SQLite's write lock in one call: ``PRAGMA busy_timeout``
set to the caller's whole remaining time, then ``BEGIN IMMEDIATE``. That call
cannot be interrupted, and SQLite counts the sleeps it asks for rather than
the time that passes, so on a machine whose sleeps overrun it lasts longer than
asked. A deferred remember waits up to 5 s while stop() waits 2 s for the
writer: shutting down while another program held memory.db raised "canonical
writer did not stop before its deadline" (seen on a GitHub macOS runner).
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from superlocalmemory.storage.write_coordinator import (
    CommandKind,
    WriteCommand,
    WriteCoordinator,
    WriteDeadlineExceededError,
    WriteResult,
)


def _coordinator(db_path: Path, monkeypatch, connection=None) -> WriteCoordinator:
    from superlocalmemory.storage.migrations import M032_write_coordinator_admission

    with sqlite3.connect(db_path) as conn:
        M032_write_coordinator_admission.apply(conn)
        conn.execute("CREATE TABLE admissions (value TEXT NOT NULL)")
    coordinator = WriteCoordinator(db_path, owner_id="waits-can-be-stopped")
    if connection is not None:
        real_open = coordinator._open_connection
        monkeypatch.setattr(coordinator, "_open_connection",
                            lambda: connection(real_open()))
    assert coordinator.claim_ownership()

    def admit(conn, _capability, command):
        conn.execute("INSERT INTO admissions(value) VALUES (?)", (command.command_id,))
        return WriteResult.from_receipt(command, {"operation_id": command.command_id})

    coordinator.register_handler(CommandKind.ADMISSION, admit)
    coordinator.start()
    return coordinator


def _command(name: str) -> WriteCommand:
    return WriteCommand.create(CommandKind.ADMISSION, {
        "journal_id": f"journal:{name}", "request_hash": f"hash:{name}",
        "profile_id": "default", "idempotency_key": f"idempotency:{name}"})


def _hold_write_lock(db_path: Path) -> sqlite3.Connection:
    blocker = sqlite3.connect(db_path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    return blocker


def _wait_until_waiting(coordinator: WriteCoordinator) -> None:
    deadline = time.monotonic() + 2.0
    while coordinator._inflight_started_at is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert coordinator._inflight_started_at is not None, "the write never started"
    time.sleep(0.2)  # well inside its wait for the lock


def _admitted(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM admissions").fetchone()[0]


class OverrunningSleeps:
    """SQLite on a VM whose sleeps run twice as long as asked."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._busy_ms = 0

    def execute(self, sql, *args):
        if sql.startswith("PRAGMA busy_timeout="):
            self._busy_ms = int(sql.split("=", 1)[1])
        if sql != "BEGIN IMMEDIATE":
            return self._connection.execute(sql, *args)
        started = time.monotonic()
        try:
            return self._connection.execute(sql)
        except sqlite3.OperationalError:
            time.sleep(max(0.0, 2 * self._busy_ms / 1000 - (time.monotonic() - started)))
            raise

    def __getattr__(self, name):
        return getattr(self._connection, name)


def test_stop_ends_a_wait_for_another_writer(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "memory.db"
    coordinator = _coordinator(db_path, monkeypatch)
    blocker = _hold_write_lock(db_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(coordinator.submit, _command("waiting"), timeout=5.0)
            _wait_until_waiting(coordinator)
            started = time.monotonic()
            coordinator.stop(deadline_s=1.0)
            assert time.monotonic() - started < 1.0
            # Not saved, and said so as contention: the journal keeps it to replay.
            with pytest.raises(WriteDeadlineExceededError):
                waiting.result(timeout=1.0)
    finally:
        blocker.rollback()
        blocker.close()
        coordinator.release_ownership()
    assert _admitted(db_path) == 0


def test_a_wait_that_runs_long_ends_when_its_caller_gives_up(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "memory.db"
    coordinator = _coordinator(db_path, monkeypatch, connection=OverrunningSleeps)
    blocker = _hold_write_lock(db_path)
    try:
        with pytest.raises(WriteDeadlineExceededError):
            coordinator.submit(_command("given-up"), timeout=1.0)
        # The caller has gone; the writer must not still be asleep in SQLite.
        coordinator.stop(deadline_s=0.5)
    finally:
        blocker.rollback()
        blocker.close()
        coordinator.release_ownership()
    assert _admitted(db_path) == 0


def test_a_write_that_gets_the_lock_in_time_is_saved(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "memory.db"
    coordinator = _coordinator(db_path, monkeypatch)
    blocker = _hold_write_lock(db_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(coordinator.submit, _command("in-time"), timeout=3.0)
            _wait_until_waiting(coordinator)
            blocker.rollback()
            assert waiting.result(timeout=3.0).receipt["operation_id"]
    finally:
        blocker.close()
        coordinator.release_ownership()
    assert _admitted(db_path) == 1
