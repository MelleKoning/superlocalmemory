# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon's graph-metrics pass computes in a separate process, with identical rows.

On the daemon's interpreter the pass (PageRank + Louvain over 480k edges on a
22k-fact store) held the interpreter and triggered repeated 75-163 ms garbage
collections; recalls during it took 5-9 s (core/graph_metrics_process.py).
"""

from __future__ import annotations

import random

import pytest

from superlocalmemory.core import graph_metrics as gm
from superlocalmemory.storage.models import AtomicFact, EdgeType, GraphEdge, MemoryRecord

_PARENT = "m-graph-proc"


@pytest.fixture
def db(engine_with_mock_deps):
    manager = engine_with_mock_deps._db
    manager.store_memory(MemoryRecord(memory_id=_PARENT, profile_id="default",
                                      content="parent record"))
    rng = random.Random(5)
    ids = [manager.store_fact(AtomicFact(memory_id=_PARENT, profile_id="default",
                                         content=f"synthetic memory {i} in cluster {i % 4}"))
           for i in range(60)]
    for i, a in enumerate(ids):  # four dense clusters with a few bridges
        for b in ids[i + 1:]:
            same = ids.index(b) % 4 == i % 4
            if (same and rng.random() < 0.5) or (not same and rng.random() < 0.03):
                manager.store_edge(GraphEdge(source_id=a, target_id=b, edge_type=EdgeType.SEMANTIC,
                                             weight=round(rng.uniform(0.2, 1.0), 3),
                                             profile_id="default"))
    return manager


def _rows(db) -> list[tuple]:
    return sorted(tuple(r) for r in db.execute(
        "SELECT fact_id, pagerank_score, community_id, degree_centrality FROM fact_importance "
        "WHERE profile_id = 'default'"))


def test_the_child_writes_exactly_the_rows_the_in_process_pass_writes(db) -> None:
    here = gm.compute_graph_metrics(db, "default")
    assert here.ok and here.written == 60
    expected = _rows(db)
    db.execute("DELETE FROM fact_importance WHERE profile_id = 'default'")
    there = gm.compute_graph_metrics(db, "default", isolate=True)
    assert there.ok, there.summary()
    assert "computed in a separate process" in there.notes
    assert _rows(db) == expected
    assert there.communities == here.communities >= 2


def test_isolated_pass_never_computes_on_this_interpreter(db, monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("PageRank/Louvain ran on the daemon's interpreter")

    monkeypatch.setattr(gm, "_networkx_metrics", boom)
    report = gm.compute_graph_metrics(db, "default", isolate=True)
    assert report.ok, report.summary()
    assert report.written == 60


def test_the_daemon_scheduler_asks_for_the_separate_process(monkeypatch) -> None:
    import inspect

    from superlocalmemory.core import maintenance_scheduler as ms

    source = inspect.getsource(ms)
    calls = [line for line in source.splitlines() if "compute_graph_metrics(" in line
             and "import" not in line]
    assert len(calls) == 2
    startup = inspect.getsource(ms.MaintenanceScheduler._initial_graph_metrics)
    assert "isolate=True" in startup
    assert source.count("isolate=True") >= 2


def test_a_failed_child_is_an_error_not_an_in_process_rerun(db, monkeypatch) -> None:
    from superlocalmemory.core import graph_metrics_process as gp

    monkeypatch.setattr(gp, "child_compute", None)  # cannot be sent to a child
    calls: list[int] = []
    monkeypatch.setattr(gm, "_networkx_metrics", lambda *a, **k: calls.append(1) or ({}, {}))
    report = gm.compute_graph_metrics(db, "default", isolate=True)
    assert not report.ok and "graph metrics process failed" in (report.error or "")
    assert calls == []


def test_the_write_back_waits_while_a_recall_runs(db, monkeypatch) -> None:
    """Each chunk of the 22k-row write-back starts only once no recall is in flight."""
    from superlocalmemory.core import recall_gate

    seen: list[int] = []
    real = gm._short_connection

    class Spy:
        def __init__(self, db_):
            self.cm = real(db_)

        def __enter__(self):
            seen.append(recall_gate.in_flight())
            return self.cm.__enter__()

        def __exit__(self, *exc):
            return self.cm.__exit__(*exc)

    monkeypatch.setattr(gm, "_short_connection", Spy)
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        while recall_gate.in_flight():
            recall_gate.end_recall()  # the recall finishes while the writer waits

    monkeypatch.setattr(gm.time, "sleep", fake_sleep)
    recall_gate.begin_recall()
    try:
        report = gm.compute_graph_metrics(db, "default")
    finally:
        while recall_gate.in_flight():
            recall_gate.end_recall()
    assert report.ok
    assert sleeps, "the write-back never waited for the recall in flight"
    assert seen[-1] == 0
