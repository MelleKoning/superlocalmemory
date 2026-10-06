# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A graph that only grew is rebuilt beside recall; one that lost a memory is not served.

On a store still being enriched the edge and fact counts move constantly, and
every recall that noticed rebuilt the whole in-memory graph on its own clock:
15 rebuilds in 60 recalls on a 481,000-edge store, 1.5-13 s each.
"""

from __future__ import annotations

import time

import pytest

from superlocalmemory.retrieval import adjacency_refresh
from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
from tests.test_retrieval.cross_scope_fixture import REQ, build_store


@pytest.fixture()
def store(tmp_path):
    return build_store(tmp_path / "g.db", n_global=0, n_shared=0, n_denied=0)


def _add_fact(store, fid: str) -> None:
    with store.db.raw_connection() as conn:
        conn.execute("INSERT INTO memories (memory_id, profile_id, scope, content)"
                     " VALUES (?, ?, 'personal', 'x')", (f"m_{fid}", REQ))
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, scope, content,"
            " fact_type, confidence, importance, evidence_count, access_count, created_at)"
            " VALUES (?, ?, ?, 'personal', 'new', 'semantic', 0.9, 0.5, 1, 0, datetime('now'))",
            (fid, f"m_{fid}", REQ))


def _wait_for(pred, timeout=20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_a_grown_graph_is_served_while_one_rebuild_runs_beside(store, monkeypatch) -> None:
    ch = EntityGraphChannel(store.db, None)
    ch._ensure_adjacency(REQ)
    loads = {"n": 0}
    real = EntityGraphChannel._load_adjacency_from_db

    def slow_load(self, *a, **k):
        loads["n"] += 1
        time.sleep(0.5)
        return real(self, *a, **k)

    monkeypatch.setattr(EntityGraphChannel, "_load_adjacency_from_db", slow_load)
    _add_fact(store, "LNEW")
    started = time.monotonic()
    ch._ensure_adjacency(REQ)
    assert time.monotonic() - started < 0.4          # the recall did not wait
    assert "LNEW" not in ch._visible_fact_ids        # it answered from the cached copy
    assert _wait_for(lambda: "LNEW" in ch._adj_slots[(REQ, False, False)].visible_fact_ids)
    ch._ensure_adjacency(REQ)
    assert "LNEW" in ch._visible_fact_ids and loads["n"] == 1


def test_a_withheld_memory_is_never_served_from_the_cached_graph(store) -> None:
    ch = EntityGraphChannel(store.db, None)
    ch._ensure_adjacency(REQ)
    gone = store.ids("L")[0]
    assert gone in ch._visible_fact_ids
    store.db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (gone,))
    ch._ensure_adjacency(REQ)
    assert gone not in ch._visible_fact_ids          # rebuilt now, synchronously


def test_without_the_change_log_the_rebuild_stays_synchronous(store) -> None:
    from superlocalmemory.storage import fact_search_changes as changes

    for trigger in changes.trigger_names():
        store.db.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    store.db.execute(f"DROP TABLE {changes.TABLE}")
    ch = EntityGraphChannel(store.db, None)
    ch._ensure_adjacency(REQ)
    _add_fact(store, "LSYNC")
    ch._ensure_adjacency(REQ)
    assert "LSYNC" in ch._visible_fact_ids
    assert not adjacency_refresh.can_serve_stale(ch, (REQ, False, False),
                                                 ch._adj_slots[(REQ, False, False)])
