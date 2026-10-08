"""Background writes wait while a person's recall runs, one bounded batch at a time.

The graph write-back and the start-up vector repair both write in batches under
the store's write lock. Each batch waits for in-flight recalls first, so a
large repair after an upgrade or a mass erase never runs straight through them.
"""

from __future__ import annotations

import threading

import pytest

from superlocalmemory.core import recall_gate
from superlocalmemory.storage.models import AtomicFact, MemoryRecord

_PARENT = "m-yield"


@pytest.fixture
def db(engine_with_mock_deps):
    manager = engine_with_mock_deps._db
    manager.store_memory(MemoryRecord(memory_id=_PARENT, profile_id="default",
                                      content="parent record for yield tests"))
    return manager


def _rows_now(db) -> int:
    rows = db.execute("SELECT COUNT(*) AS n FROM fact_importance WHERE profile_id='default'")
    return int(dict(rows[0])["n"])


def test_stale_graph_rows_are_deleted_only_after_waiting_for_recalls(db, monkeypatch) -> None:
    from superlocalmemory.core import graph_metrics

    ids = [db.store_fact(AtomicFact(memory_id=_PARENT, profile_id="default",
                                    content=f"Fixture yield fact {i}.")) for i in range(3)]
    graph_metrics._write(db, "default", [(f, "default", 0.1, 0, 0.0, 0.0) for f in ids])
    assert _rows_now(db) == 3

    seen: list[int] = []
    monkeypatch.setattr(graph_metrics, "_yield_to_recalls", lambda: seen.append(_rows_now(db)))
    graph_metrics._write(db, "default", [])  # every row is now stale

    assert seen, "the stale-row deletes never waited for recalls"
    assert seen[0] == 3  # the wait came before the first delete
    assert _rows_now(db) == 0


class _Store:
    """A vector index that records how many recalls were in flight at each write."""

    def __init__(self, on_first) -> None:
        self.seen: list[int] = []
        self._on_first = on_first

    def upsert(self, fact_id, profile_id, embedding) -> bool:
        self.seen.append(recall_gate.in_flight())
        if len(self.seen) == 1:
            self._on_first()
        return True


def test_each_vector_repair_batch_waits_for_recalls(db) -> None:
    from superlocalmemory.server.vector_backfill import upsert_missing

    base = recall_gate.in_flight()
    timers: list[threading.Timer] = []

    def recall_for(seconds: float) -> None:
        hold = recall_gate.RecallHold()
        timer = threading.Timer(seconds, hold.leave)
        timers.append(timer)
        timer.start()

    # A recall is running when the repair starts, and another one arrives
    # while the first batch is being written.
    recall_for(0.5)
    store = _Store(on_first=lambda: recall_for(0.5))
    missing = [(f"f{i}", "default", [0.1, 0.2]) for i in range(4)]
    written = upsert_missing(db, store, "default", missing, batch=2, pause=0.0)
    for timer in timers:
        timer.join(10)

    assert written == 4
    # Batch one (writes 1-2) started after the first recall ended; batch two
    # (writes 3-4) after the recall that arrived during batch one.
    assert store.seen[0] == base and store.seen[2] == base
    assert recall_gate.in_flight() == base


def test_an_older_graph_pass_never_deletes_a_newer_passes_rows(db, monkeypatch) -> None:
    """Pass A reads the graph, stalls; F is stored; pass B runs. A's write
    must not delete the row B wrote for F."""
    from superlocalmemory.core import graph_metrics

    def store(name: str) -> str:
        return db.store_fact(AtomicFact(memory_id=_PARENT, profile_id="default",
                                        content=f"Fixture pass fact {name}."))

    store("one")
    store("two")
    in_compute, release = threading.Event(), threading.Event()
    real = graph_metrics._networkx_metrics
    calls: list[int] = []

    def stalled_first(edges, damping):
        calls.append(1)
        if len(calls) == 1:
            in_compute.set()
            assert release.wait(30)
        return real(edges, damping)

    monkeypatch.setattr(graph_metrics, "_networkx_metrics", stalled_first)
    reports: dict[str, object] = {}
    pass_a = threading.Thread(target=lambda: reports.update(
        a=graph_metrics.compute_graph_metrics(db, "default")))
    pass_a.start()
    assert in_compute.wait(10)
    fresh = store("stored-between-passes")
    pass_b = threading.Thread(target=lambda: reports.update(
        b=graph_metrics.compute_graph_metrics(db, "default")))
    pass_b.start()
    pass_b.join(5)  # without one-pass-at-a-time, B finishes here
    release.set()
    pass_a.join(30)
    pass_b.join(30)

    assert reports["a"].error is None and reports["b"].error is None
    rows = db.execute("SELECT fact_id FROM fact_importance WHERE profile_id='default'")
    assert fresh in {dict(r)["fact_id"] for r in rows}, "a newer pass's row was deleted"
