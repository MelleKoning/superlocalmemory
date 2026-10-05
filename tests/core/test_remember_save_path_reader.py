# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A save never waits on a journal reader.

Under a burst the journal's writer commits every few milliseconds. A reader
that starts a read transaction in that window is made by SQLite to retry for
its WAL snapshot with growing sleeps, which stalled saves for seconds until
they ran out of budget and were refused as "too many saves". That refusal is
what test_386 recorded in 2 of 10 loaded runs before 4.1.20, and what was
misread at the time as a legacy child process exiting.

This drives the whole remember path the stress test uses (runtime, admission,
journal, canonical writer) and fails the moment any step borrows a journal
reader on the caller's thread, so the stall cannot come back through a new
read anywhere along that path.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
from superlocalmemory.storage.admission_journal import Actor, RememberRequest

_ACTOR = "save-path-reader-daemon"
_DEADLINE_MS = 2_000
_ACCEPT_AFTER_MS = 1_200
_SAVES = 8


def _build(data_dir: Path) -> CanonicalRememberRuntime:
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    db = DatabaseManager(data_dir / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        for migration in (
            M018_ingestion_operations,
            M032_write_coordinator_admission,
            M033_projection_transactions,
            M034_obligation_integrity,
        ):
            migration.apply(conn)
    return CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=build_immediate_admission_handler(db, profile_id="default"),
        journal_path=data_dir / "admission_journal.db",
        owner_id="save-path-reader-runtime",
    )


def _request(sequence: int) -> RememberRequest:
    return RememberRequest(
        content=f"Save path fact {sequence}: written without a journal reader.",
        profile_id="default",
        source_type="save-path-reader",
        idempotency_key=f"save-path-reader:{sequence}",
        trusted_actor_id=_ACTOR,
    )


def _actor() -> Actor:
    return Actor(_ACTOR, frozenset({"default"}), frozenset({"personal"}))


@pytest.fixture()
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "slm-save-path-reader"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    rt = _build(data_dir)
    rt.start()
    yield rt
    rt.stop()


def test_a_save_and_its_retry_never_borrow_a_journal_reader(
    runtime: CanonicalRememberRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = runtime.journal
    original = journal._read_connection
    caller = threading.current_thread()
    borrowed: list[str] = []

    def guarded_reader(*args, **kwargs):
        # Background work (startup replay, a deferred commit) may read; the
        # caller's save must not.
        if threading.current_thread() is caller:
            borrowed.append("journal reader borrowed on the save path")
            raise AssertionError(borrowed[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(journal, "_read_connection", guarded_reader)

    statuses = []
    for sequence in range(_SAVES):
        for _attempt in ("first", "idempotent retry"):
            receipt = runtime.remember(
                _request(sequence),
                _actor(),
                deadline_ms=_DEADLINE_MS,
                accept_after_ms=_ACCEPT_AFTER_MS,
            )
            statuses.append(receipt.payload["status"])

    assert borrowed == []
    assert set(statuses) <= {"queryable", "accepted"}
    assert runtime.wait_for_deferred(timeout=30.0)
    monkeypatch.setattr(journal, "_read_connection", original)
    assert journal.count() == _SAVES
