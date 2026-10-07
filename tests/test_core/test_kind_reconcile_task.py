# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The background check of confirmed kinds: reaches every drifted row, runs
once per store at a time, is not repeated while nothing could have drifted,
and stops when its engine closes.

On a copy of a 2 GB store the check read for 22 s on every start of every
process that opened the store (daemon, MCP server, recall worker, each CLI
command), inside the start-up path.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from superlocalmemory.core.kind_reconcile_task import KindReconcileTask, _profile_key
from superlocalmemory.storage import memory_kind_writes as writes
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    from superlocalmemory.storage.migrations import M052_memory_kinds as m052

    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    with mgr.raw_connection() as conn:
        m052.apply(conn)
    mgr._kind_columns_present = True  # noqa: SLF001 - test shortcut
    return mgr


def _confirmed(db: DatabaseManager, kind: str, fact_type: str) -> str:
    mid = db.store_memory(MemoryRecord(profile_id="default", content="session"))
    fid = db.store_fact(AtomicFact(profile_id="default", memory_id=mid,
                                   content=f"fixture {kind} {uuid.uuid4().hex}",
                                   fact_type=FactType.SEMANTIC))
    db.execute("UPDATE atomic_facts SET memory_kind = ?, memory_kind_source = 'user', "
               "fact_type = ? WHERE fact_id = ?", (kind, fact_type, fid))
    return fid


def _fact_type(db: DatabaseManager, fid: str) -> str:
    return dict(db.execute("SELECT fact_type FROM atomic_facts WHERE fact_id = ?", (fid,))[0])[
        "fact_type"]


def _run(task: KindReconcileTask) -> dict:
    assert task.start().wait(timeout=30)
    return task.snapshot()


def test_a_drifted_row_past_the_first_limit_confirmed_rows_is_repaired(db) -> None:
    """The LIMIT used to apply to every confirmed row: a store with more than
    ``limit`` of them only ever looked at the same first ones, so a drifted
    row after them was never repaired, on any start."""
    for _ in range(3):
        _confirmed(db, "decision", "episodic")  # already in step
    drifted = _confirmed(db, "decision", "semantic")
    assert db.execute("SELECT COUNT(*) AS n FROM atomic_facts WHERE memory_kind_source = 'user'"
                      )[0]["n"] == 4
    assert writes.reconcile_confirmed(db, "default", limit=2) == 1
    assert _fact_type(db, drifted) == "episodic"


def test_a_pass_repairs_in_short_turns_and_reports_them(db, monkeypatch) -> None:
    rows = [_confirmed(db, "decision", "semantic") for _ in range(4)]
    clock = iter(range(1_000))  # every clock read is one "second" later
    monkeypatch.setattr(writes.time, "perf_counter", lambda: float(next(clock)))
    result = writes.reconcile_confirmed_pass(db, "default", max_seconds=1.5, pause_seconds=0)
    assert result.fixed == 4 and result.complete is True
    assert result.transactions >= 2  # the budget split the work, nothing was skipped
    assert all(_fact_type(db, fid) == "episodic" for fid in rows)


def test_a_finished_pass_is_not_repeated_while_the_same_version_runs(db) -> None:
    drifted = _confirmed(db, "decision", "semantic")
    first = _run(KindReconcileTask(db, "default", pause_seconds=0))
    assert first["state"] == "complete" and first["fixed"] == 1
    assert _fact_type(db, drifted) == "episodic"
    again = _run(KindReconcileTask(db, "default", pause_seconds=0))
    assert again["state"] == "up_to_date"  # no second 22 s read


def test_another_version_starting_makes_the_next_start_check_again(db) -> None:
    assert _run(KindReconcileTask(db, "default", pause_seconds=0))["state"] == "complete"
    drifted = _confirmed(db, "decision", "semantic")  # written by the older version
    (db.db_path.parent / ".last_version").write_text("4.1.18", encoding="utf-8")
    again = _run(KindReconcileTask(db, "default", pause_seconds=0))
    assert again["state"] == "complete" and again["fixed"] == 1
    assert _fact_type(db, drifted) == "episodic"


def test_a_second_engine_observes_a_check_already_running(db) -> None:
    from superlocalmemory.core.file_lock import exclusive_lock

    _confirmed(db, "decision", "semantic")
    lock = db.db_path.parent / f".kind-reconcile-{_profile_key('default')}.lock"
    with exclusive_lock(lock, timeout_s=0.0):  # another engine's pass holds it
        state = _run(KindReconcileTask(db, "default", pause_seconds=0))
    assert state["state"] == "observed"


def test_stopping_interrupts_the_read_itself(db, monkeypatch) -> None:
    _confirmed(db, "decision", "semantic")
    monkeypatch.setattr(writes, "_STOP_POLL_OPS", 1)
    with pytest.raises(writes.ReconcileStopped):
        writes.reconcile_confirmed_pass(db, "default", should_stop=lambda: True)
