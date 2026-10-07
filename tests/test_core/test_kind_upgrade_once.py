# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""An upgraded store's older memories are classified once per profile, by itself.

Never twice: a profile with any earlier run (running, paused, completed,
cancelled, undone) is observed or left alone; two processes starting at once
queue one run between them. Never off the device without confirmation.
And a batch's time spent WAITING for the write lock is no longer reported,
or paced, as time it HELD the lock.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from superlocalmemory.core import kind_upgrade
from superlocalmemory.core.memory_kind_backfill_plan import BackendChoice
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.memory_kind_store import MemoryKindStore
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

RULES = BackendChoice("auto", "rules", "Typed on this device by simple rules.", False)
ONLINE = BackendChoice("jev", "jev", "online", True)


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    from superlocalmemory.storage.migrations import M052_memory_kinds as m052

    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    with mgr.raw_connection() as conn:
        m052.apply(conn)
    mgr._kind_columns_present = True  # noqa: SLF001 - test shortcut
    return mgr


def _legacy_facts(db: DatabaseManager, n: int, profile: str = "default") -> None:
    mid = db.store_memory(MemoryRecord(profile_id=profile, content="session"))
    for _ in range(n):
        fid = db.store_fact(AtomicFact(profile_id=profile, memory_id=mid,
                                       content=f"older note {uuid.uuid4().hex}",
                                       fact_type=FactType.SEMANTIC))
        db.execute("UPDATE atomic_facts SET memory_kind = NULL, memory_kind_source = NULL "
                   "WHERE fact_id = ?", (fid,))


def _runs(db: DatabaseManager) -> list[dict]:
    return [dict(r) for r in db.execute("SELECT * FROM memory_kind_runs ORDER BY created_at")]


def test_an_untyped_store_is_queued_once_and_then_only_observed(db) -> None:
    _legacy_facts(db, 3)
    store = MemoryKindStore(db)
    first = kind_upgrade.ensure_profile_typed(db, store, RULES, True, "default")
    assert first["state"] == "queued" and first["untyped"] == 3
    second = kind_upgrade.ensure_profile_typed(db, store, RULES, True, "default")
    assert second == {**second, "state": "observed", "run_id": first["run_id"]}
    assert len(_runs(db)) == 1
    assert _runs(db)[0]["requested_by"] == kind_upgrade.REQUESTED_BY


@pytest.mark.parametrize("status", ["completed", "cancelled", "reverted", "failed"])
def test_a_profile_that_had_a_run_is_never_classified_again_by_itself(db, status) -> None:
    """The M5 store's completed run 438a2e8d2c034028 is this case."""
    _legacy_facts(db, 2)
    store = MemoryKindStore(db)
    run = store.create_run("default", backend="rules", recipe_id="kinds-rules-v1",
                           mode="untyped", requested_by="person", total_estimate=2)
    db.execute("UPDATE memory_kind_runs SET status = ? WHERE run_id = ?", (status, run["run_id"]))
    _legacy_facts(db, 2)  # still untyped facts: it is the person's decision, not a gap
    got = kind_upgrade.ensure_profile_typed(db, store, RULES, True, "default")
    assert got["state"] == "already_classified" and got["run_id"] == run["run_id"]
    assert len(_runs(db)) == 1


def test_two_processes_starting_at_once_queue_one_run(tmp_path, db) -> None:
    _legacy_facts(db, 2)
    barrier, results = threading.Barrier(4), []

    def start() -> None:
        own = DatabaseManager(db.db_path)  # each "process" has its own manager
        own._kind_columns_present = True  # noqa: SLF001
        barrier.wait()
        results.append(MemoryKindStore(own).create_run_once(
            "default", backend="rules", recipe_id="kinds-rules-v1",
            requested_by=kind_upgrade.REQUESTED_BY, total_estimate=2))

    threads = [threading.Thread(target=start) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sum(r is not None for r in results) == 1
    assert len(_runs(db)) == 1


def test_never_off_the_device_and_never_when_nothing_is_untyped(db) -> None:
    store = MemoryKindStore(db)
    assert kind_upgrade.ensure_profile_typed(db, store, RULES, True, "default")["state"] \
        == "not_needed"
    _legacy_facts(db, 2)
    assert kind_upgrade.ensure_profile_typed(db, store, ONLINE, True, "default")["state"] \
        == "needs_confirmation"
    assert kind_upgrade.ensure_profile_typed(db, store, RULES, False, "default")["state"] \
        == "disabled"
    assert _runs(db) == []


def test_the_runner_queues_and_finishes_it_by_itself(db) -> None:
    from superlocalmemory.core.memory_kind_backfill import BackfillRunner

    _legacy_facts(db, 5)

    class _Engine:
        _db = db
        _config = None

    runner = BackfillRunner(engine_supplier=lambda: _Engine(), store=MemoryKindStore(db))
    runner._choice = lambda engine, cfg: RULES  # noqa: SLF001 - on-device rules backend
    runner._materializer_due = lambda _db: False  # noqa: SLF001
    runner.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not any(
                r["status"] == "completed" for r in _runs(db)):
            time.sleep(0.05)
    finally:
        assert runner.stop(timeout_s=10)
    assert [r["status"] for r in _runs(db)] == ["completed"]
    assert runner.status("default")["automatic_classification"]["state"] == "queued"
    assert db.execute("SELECT COUNT(*) AS n FROM atomic_facts WHERE memory_kind IS NULL")[0][
        "n"] == 0
    restarted = BackfillRunner(engine_supplier=lambda: _Engine(), store=MemoryKindStore(db))
    restarted._choice = lambda engine, cfg: RULES  # noqa: SLF001
    restarted._ensure_typed_once()  # noqa: SLF001 - what a restart does first
    assert restarted.upgrade["default"]["state"] == "already_classified"
    assert len(_runs(db)) == 1


def test_untyped_zero_with_older_types_left_is_explained(db) -> None:
    from superlocalmemory.core.memory_kind_runs import explain_counts

    out = explain_counts({"schema_ready": True,
                          "counts": {"untyped": 0, "legacy": 52, "legacy_no_kind": 0}})
    assert "52" in out["explanation"] and "not confident enough" in out["explanation"]
    out = explain_counts({"schema_ready": True,
                          "counts": {"untyped": 0, "legacy": 9, "legacy_no_kind": 9}})
    assert "no kind of their own yet" in out["explanation"]


def test_waiting_for_the_lock_is_not_counted_as_holding_it(db) -> None:
    """The 4.1.21 run's max batch 'write' of 6.894 s was time spent waiting."""
    from superlocalmemory.storage.memory_kind_writes import revert_batch

    run = MemoryKindStore(db).create_run("default", backend="rules", recipe_id="r",
                                         mode="untyped", requested_by="t", total_estimate=0)
    db.execute("UPDATE memory_kind_runs SET status='reverting' WHERE run_id=?", (run["run_id"],))
    other = sqlite3.connect(db.db_path, isolation_level=None, check_same_thread=False)
    other.execute("BEGIN IMMEDIATE")
    release = threading.Timer(1.0, lambda: other.execute("ROLLBACK"))
    release.start()
    try:
        result = revert_batch(db, run["run_id"], "default", limit=10)
    finally:
        release.join()
        other.close()
    assert result.wait_ms >= 800
    assert result.write_ms < 500


def test_a_run_finishing_between_the_check_and_the_queue_still_counts(db) -> None:
    """The guard is inside the one INSERT: a run that completed after the
    check read the runs table still stops the automatic one."""
    store = MemoryKindStore(db)
    run = store.create_run("default", backend="rules", recipe_id="kinds-rules-v1",
                           mode="untyped", requested_by="person", total_estimate=0)
    db.execute("UPDATE memory_kind_runs SET status='completed' WHERE run_id=?", (run["run_id"],))
    assert store.create_run_once("default", backend="rules", recipe_id="kinds-rules-v1",
                                 requested_by=kind_upgrade.REQUESTED_BY,
                                 total_estimate=1) is None
    assert len(_runs(db)) == 1


def test_under_a_backlog_the_pause_follows_the_measured_lock_hold(db) -> None:
    """New memories still go first, but a batch that held the lock 30 ms waits
    ~0.6 s for its turn, not a flat 10 s (21,500 rows took 81 minutes)."""
    from superlocalmemory.core.memory_kind_backfill import BackfillRunner

    _legacy_facts(db, 120)
    store = MemoryKindStore(db)
    store.create_run("default", backend="rules", recipe_id="kinds-rules-v1", mode="untyped",
                     requested_by="t", total_estimate=120)
    now = [100.0]  # a monotonic clock is never 0

    class _Engine:
        _db = db
        _config = None

    runner = BackfillRunner(engine_supplier=lambda: _Engine(), store=store,
                            clock=lambda: now[0])
    runner._choice = lambda engine, cfg: RULES  # noqa: SLF001
    runner._materializer_due = lambda _db: True  # noqa: SLF001 - ingestion never stops
    runner._last_hold_s = 0.03  # noqa: SLF001 - as measured by the previous batch
    assert runner.run_once().action == "yield"
    now[0] += 0.7
    assert runner.run_once().action == "batch"
    assert runner.batch_write_ms and runner.batch_wait_ms  # both measured, separately
