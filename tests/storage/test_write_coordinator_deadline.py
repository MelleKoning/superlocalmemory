# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A foreground write waits for another writer exactly as long as its deadline allows."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from tests.storage.test_write_coordinator import _admission_payload, _install_write_commits


class _SteppedClock:
    """``time`` for the coordinator, whose monotonic clock moves only when the
    test moves it: one step per ``BEGIN IMMEDIATE`` attempt. Whether a write
    is inside its deadline is then decided by how many times it had to wait,
    never by how fast the machine is (a slow Windows runner used to run the
    real 1.8 s deadline out and fail)."""

    def __init__(self) -> None:
        import time as _real

        self._real = _real
        self._lock = threading.Lock()
        self.now = 1_000.0

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> float:
        with self._lock:
            self.now += seconds
            return self.now

    def __getattr__(self, name):  # sleep, time, ... stay real
        return getattr(self._real, name)


def _contended_write(tmp_path, monkeypatch, *, timeout, step, release_after):
    """Submit one admission while a legacy connection holds the write lock.

    Every BEGIN IMMEDIATE the worker attempts moves the coordinator's clock by
    ``step``. Once it has moved ``release_after`` past the submit, the legacy
    writer lets go, and the attempt after that is let through only once the
    lock really is free. ``release_after=None`` never lets go. Returns the
    outcome (a WriteResult or the raised error) and the clock at the BEGIN
    that succeeded (None if none did).
    """
    from superlocalmemory.storage import write_coordinator as wc
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path)
    clock = _SteppedClock()
    release_blocker = threading.Event()
    blocker_released = threading.Event()
    began_at: list[float] = []
    real_open_connection = coordinator._open_connection

    class SteppingConnection:
        def __init__(self, connection) -> None:
            self._connection = connection
            self._submitted_at = clock.monotonic()

        def execute(self, sql, *args):
            if sql != "BEGIN IMMEDIATE":
                return self._connection.execute(sql, *args)
            if release_blocker.is_set():
                # Wait for the lock to be really gone, so the next attempt is
                # decided by the deadline alone, not by a race with rollback.
                assert blocker_released.wait(timeout=30)
            else:
                now = clock.advance(step)
                if release_after is not None and now - self._submitted_at > release_after:
                    release_blocker.set()
                    assert blocker_released.wait(timeout=30)
            result = self._connection.execute(sql, *args)
            began_at.append(clock.monotonic() - self._submitted_at)
            return result

        def __getattr__(self, name):
            return getattr(self._connection, name)

    monkeypatch.setattr(wc, "time", clock)
    monkeypatch.setattr(
        coordinator, "_open_connection",
        lambda: SteppingConnection(real_open_connection()),
    )
    assert coordinator.claim_ownership()
    coordinator.register_handler(
        CommandKind.ADMISSION,
        lambda _conn, _capability, command: WriteResult.from_receipt(
            command, {"operation_id": "operation:deadline-contention"},
        ),
    )
    coordinator.start()
    blocker_ready = threading.Event()

    def hold_legacy_write() -> None:
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            blocker_ready.set()
            release_blocker.wait(timeout=60)
            connection.rollback()
        finally:
            connection.close()
            blocker_released.set()

    blocker = threading.Thread(target=hold_legacy_write)
    blocker.start()
    assert blocker_ready.wait(timeout=30)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                coordinator.submit,
                WriteCommand.create(CommandKind.ADMISSION,
                                    _admission_payload("deadline-contention")),
                timeout=timeout,
            )
            try:
                outcome = future.result(timeout=60)  # a hang guard, not a margin
            except Exception as exc:  # noqa: BLE001 - the outcome under test
                outcome = exc
    finally:
        release_blocker.set()
        blocker.join(timeout=30)
        coordinator.release_ownership()
    assert not blocker.is_alive()
    return outcome, (began_at[0] if began_at else None)


def test_foreground_command_uses_its_remaining_deadline_for_sqlite_contention(
    tmp_path,
    monkeypatch,
) -> None:
    """A legacy writer may release after SQLite's old fixed one-second wait.

    The worker keeps waiting for as long as the caller's deadline allows, so
    a lock held 1.2 s is waited out, not reported busy. The budget is large:
    the caller's own wait runs on the real clock, and must never be the
    thing that ends this on a slow machine.
    """
    outcome, began = _contended_write(tmp_path, monkeypatch,
                                      timeout=60.0, step=0.25, release_after=1.2)
    assert not isinstance(outcome, Exception), repr(outcome)
    assert outcome.receipt["operation_id"] == "operation:deadline-contention"
    assert began is not None and began > 1.0, "the write must wait past the old one second"


def test_a_lock_held_past_the_deadline_expires_the_write(tmp_path, monkeypatch) -> None:
    """The same wait ends at the caller's deadline with a deadline error,
    raised by the worker's own clock after three waits of 20 s each, not by
    the caller giving up on the real clock."""
    from superlocalmemory.storage.write_coordinator import WriteDeadlineExceededError

    outcome, began = _contended_write(tmp_path, monkeypatch,
                                      timeout=60.0, step=20.0, release_after=None)
    assert isinstance(outcome, WriteDeadlineExceededError), repr(outcome)
    assert "waiting for another writer" in str(outcome)
    assert began is None
