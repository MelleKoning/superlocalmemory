# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Background reads never hold the store's write lock, so edits and saves go on.

Measured on a copy of a 22k-fact store (4.1.21): the graph-metrics pass read
every edge through ``raw_connection``, which takes the process-wide write lock,
and held it 60.5 s; the start-up kind check held it 33.5 s; a recall's
correction check waited 44 s behind them; and an edit's event write waited on
the daemon's request loop for 60 s, freezing every request. Each test here
holds the lock in another thread (as a long writer would) and requires the
read to finish anyway.
"""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.storage.models import AtomicFact, EdgeType, GraphEdge, MemoryRecord

_PARENT = "m-lock-holds"


@pytest.fixture
def db(engine_with_mock_deps):
    manager = engine_with_mock_deps._db
    manager.store_memory(MemoryRecord(memory_id=_PARENT, profile_id="default",
                                      content="parent record for lock-hold tests"))
    return manager


def _facts(db, n: int) -> list[str]:
    return [db.store_fact(AtomicFact(memory_id=_PARENT, profile_id="default",
                                     content=f"Fixture lock fact {i} about node N{i}."))
            for i in range(n)]


class _LockHeld:
    """Hold ``lock`` in another thread, like a long background write."""

    def __init__(self, lock) -> None:
        self._lock, self._taken, self._release = lock, threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        with self._lock:
            self._taken.set()
            self._release.wait(timeout=30)

    def __enter__(self):
        self._thread.start()
        assert self._taken.wait(timeout=5)
        return self

    def __exit__(self, *exc) -> None:
        self._release.set()
        self._thread.join(timeout=5)


def _finishes_within(seconds: float, work) -> object:
    box: dict = {}
    worker = threading.Thread(target=lambda: box.update(out=work()), daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(timeout=seconds)
    assert not worker.is_alive(), f"blocked behind the write lock for > {seconds}s"
    assert time.monotonic() - started < seconds
    return box.get("out")


def test_the_graph_metrics_staleness_check_reads_without_the_write_lock(db) -> None:
    from superlocalmemory.core.graph_metrics import metrics_are_stale

    _facts(db, 3)
    with _LockHeld(db._lock):
        stale, _why = _finishes_within(3.0, lambda: metrics_are_stale(db, "default"))
    assert stale is True


def test_the_graph_metrics_pass_reads_the_whole_graph_without_the_write_lock(
    db, monkeypatch,
) -> None:
    from superlocalmemory.core import graph_metrics

    ids = _facts(db, 4)
    db.store_edge(GraphEdge(profile_id="default", source_id=ids[0], target_id=ids[1],
                            edge_type=EdgeType.ENTITY, weight=1.0))
    reading, release = threading.Event(), threading.Event()
    real = graph_metrics.iter_logical_edges

    def slow_read(conn, profile_id):
        reading.set()
        assert release.wait(timeout=10)  # the whole-graph read is in progress
        yield from real(conn, profile_id)

    monkeypatch.setattr(graph_metrics, "iter_logical_edges", slow_read)
    runner = threading.Thread(
        target=lambda: graph_metrics.compute_graph_metrics(db, "default"), daemon=True)
    runner.start()
    assert reading.wait(timeout=5)
    try:
        # An interactive write while the pass is reading every edge.
        _finishes_within(3.0, lambda: db.execute(
            "UPDATE atomic_facts SET access_count = access_count + 1 WHERE fact_id = ?",
            (ids[2],)))
    finally:
        release.set()
        runner.join(timeout=30)


def test_the_metrics_are_written_a_bounded_chunk_per_transaction(db, monkeypatch) -> None:
    from superlocalmemory.core import graph_metrics

    entered: list[int] = []
    real = graph_metrics._short_connection

    def counting(manager):
        entered.append(1)
        return real(manager)

    monkeypatch.setattr(graph_metrics, "_short_connection", counting)
    monkeypatch.setattr(graph_metrics, "_WRITE_CHUNK", 2)
    rows = [(fid, "default", 0.1, 0, 0.0, 0.0) for fid in _facts(db, 5)]
    graph_metrics._write(db, "default", rows)

    # One for the column check, then one per chunk: never one for the table.
    assert len(entered) == 1 + 3
    count = db.execute("SELECT COUNT(*) AS n FROM fact_importance WHERE profile_id='default'")
    assert dict(count[0])["n"] == len(rows)


def test_a_recall_correction_check_reads_without_the_write_lock(db) -> None:
    ids = _facts(db, 2)
    with _LockHeld(db._lock):
        out = _finishes_within(3.0, lambda: db.get_correction_inadmissible_fact_ids(
            ids, "default"))
    assert out == set()


def test_the_start_up_kind_check_takes_no_write_transaction_when_nothing_needs_repair(
    db, monkeypatch,
) -> None:
    from superlocalmemory.storage.memory_kind_writes import reconcile_confirmed

    if not db.has_memory_kind_columns():
        pytest.skip("store without kind columns")
    fid = _facts(db, 1)[0]
    db.execute("UPDATE atomic_facts SET memory_kind='decision', memory_kind_source='user', "
               "fact_type='episodic' WHERE fact_id=?", (fid,))
    with _LockHeld(db._lock):
        fixed = _finishes_within(3.0, lambda: reconcile_confirmed(db, "default"))
    assert fixed == 0


def test_the_start_up_kind_check_still_repairs_a_drifted_row(db) -> None:
    from superlocalmemory.storage.memory_kind_writes import reconcile_confirmed

    if not db.has_memory_kind_columns():
        pytest.skip("store without kind columns")
    fid = _facts(db, 1)[0]
    db.execute("UPDATE atomic_facts SET memory_kind='decision', memory_kind_source='user', "
               "fact_type='semantic' WHERE fact_id=?", (fid,))
    assert reconcile_confirmed(db, "default") == 1
    row = db.execute("SELECT fact_type FROM atomic_facts WHERE fact_id=?", (fid,))[0]
    assert dict(row)["fact_type"] == "episodic"


def test_an_event_waits_a_bounded_time_for_a_busy_store(tmp_path) -> None:
    from superlocalmemory.infra import event_bus
    from superlocalmemory.storage.write_lock import get_write_lock

    path = tmp_path / "events.db"
    event_bus.EventBus.reset_instance(path)
    bus = event_bus.EventBus.get_instance(path)
    received: list[dict] = []
    bus.subscribe(received.append)
    with _LockHeld(get_write_lock(path)):
        _finishes_within(event_bus._PERSIST_LOCK_WAIT_S + 2.0,
                         lambda: bus.emit("memory.stored", {"fact_id": "f1"}))
    # Not stored while the store was busy, but still delivered live.
    assert [e["event_type"] for e in received] == ["memory.stored"]


@pytest.fixture
def make_engine():
    made: list = []

    def _make(*args, **kwargs):
        from superlocalmemory.core.engine import MemoryEngine

        made.append(MemoryEngine(*args, **kwargs))
        return made[-1]

    yield _make
    for engine in made:
        engine.close()


def test_the_start_up_kind_check_never_holds_up_engine_start(make_engine, mode_a_config,
                                                              monkeypatch) -> None:
    """A real engine start returns while its kind check is still reading."""
    from superlocalmemory.core.engine import Capabilities
    from superlocalmemory.storage import memory_kind_writes

    release = threading.Event()
    real_pass = memory_kind_writes.reconcile_confirmed_pass

    def slow_pass(db, profile_id, **kw):
        assert release.wait(timeout=10)  # a 2 GB store: the read takes a while
        return real_pass(db, profile_id, **kw)

    monkeypatch.setattr(memory_kind_writes, "reconcile_confirmed_pass", slow_pass)
    engine = make_engine(mode_a_config, capabilities=Capabilities.LIGHT)
    engine.initialize()  # returned while the pass is still blocked (no clock bound:
    task = engine._kind_reconcile  # a start that waited would see it fail instead)
    assert not release.is_set()
    assert task.snapshot()["state"] in ("pending", "running")  # truthful while it runs
    release.set()
    assert task.wait(timeout=10) and task.snapshot()["state"] == "complete"


def test_closing_the_engine_stops_its_kind_check_and_joins_it(make_engine, mode_a_config,
                                                              monkeypatch) -> None:
    """No thread keeps reading a store after its engine closed it."""
    from superlocalmemory.core.engine import Capabilities
    from superlocalmemory.storage import memory_kind_writes

    def stuck_pass(db, profile_id, *, should_stop, **_kw):
        while not should_stop():
            time.sleep(0.01)
        raise memory_kind_writes.ReconcileStopped("stopped")

    monkeypatch.setattr(memory_kind_writes, "reconcile_confirmed_pass", stuck_pass)
    engine = make_engine(mode_a_config, capabilities=Capabilities.LIGHT)
    engine.initialize()
    task = engine._kind_reconcile
    engine.close()
    assert task.wait(timeout=5) and task.snapshot()["state"] == "stopped"
