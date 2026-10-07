# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""No recall waits on somebody else's entity-graph build.

Measured on a copy of a 21,738-fact store: the first recall after the daemon
reported ready took 8.5-13 s while its channels took 1.2 s; ~6 s was spent
queued on the entity graph's cache lock while the start-up warm-up built the
graph under it. Now (retrieval/adjacency_rcu, retrieval/entity_graph_warmup):

* builds run with the cache lock released and swap the finished graph in, so a
  recall whose scope has a cached graph uses it while another scope builds;
* a recall whose scope has no usable graph while another thread builds it
  reports ``entity_graph`` as ``warming`` (incomplete) instead of waiting, and
  gets the boost back once the graph is in;
* with no build in flight, a recall builds the graph itself, as before;
* the warm-up builds the graph before its own recalls, and /health says what
  it is still building.

Real ``EntityGraphChannel`` on a real SQLite store; waits are event-based.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from superlocalmemory.retrieval import adjacency_rcu
from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval import entity_graph_warmup as egw
from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
from superlocalmemory.server import recall_warmup
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager

PROFILES = ("a", "b")
QUERY = "Alice met Bob"


def _channel(tmp_path) -> EntityGraphChannel:
    db = DatabaseManager(tmp_path / "warm_graph.db")
    with db.raw_connection() as conn:
        schema.create_all_tables(conn)
        for pid in PROFILES:
            conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                         (pid, pid))
            conn.execute(
                "INSERT INTO memories (memory_id, profile_id, scope, shared_with, content) "
                "VALUES (?, ?, 'personal', NULL, ?)", (f"m_{pid}", pid, "note"))
            conn.execute(
                "INSERT INTO atomic_facts "
                "(fact_id, memory_id, profile_id, scope, shared_with, content, "
                " fact_type, confidence, importance, evidence_count, access_count, "
                " canonical_entities_json, embedding, created_at) "
                "VALUES (?, ?, ?, 'personal', NULL, 'note', 'semantic', 0.9, 0.5, 1, 0, "
                "'[]', ?, datetime('now'))",
                (f"f_{pid}", f"m_{pid}", pid, json.dumps([1.0, 0.0, 0.0, 0.0])))
        conn.commit()
    return EntityGraphChannel(db)


class _BlockedBuild:
    """Make one profile's graph build stop half-way until released."""

    def __init__(self, channel, profile_id, monkeypatch) -> None:
        self.inside, self.release = threading.Event(), threading.Event()
        real = EntityGraphChannel._load_adjacency_from_db

        def load(this, pid, **kw):
            if pid == profile_id:
                self.inside.set()
                assert self.release.wait(10), "test never released the build"
            return real(this, pid, **kw)

        monkeypatch.setattr(EntityGraphChannel, "_load_adjacency_from_db", load)
        self.thread = threading.Thread(target=self._build, args=(channel, profile_id),
                                       daemon=True)

    @staticmethod
    def _build(channel, profile_id) -> None:
        with channel._cache_lock:
            channel._ensure_adjacency(profile_id)

    def __enter__(self):
        self.thread.start()
        assert self.inside.wait(5)
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(10)
        return False


def test_a_cached_scope_is_served_while_another_scope_builds(tmp_path, monkeypatch):
    channel = _channel(tmp_path)
    assert isinstance(egw.score_candidates_unless_warming(channel, QUERY, ["f_a"], "a"), dict)
    with _BlockedBuild(channel, "b", monkeypatch) as build:
        got = {}
        t = threading.Thread(target=lambda: got.update(
            r=egw.score_candidates_unless_warming(channel, QUERY, ["f_a"], "a")))
        t.start()
        t.join(3)  # a reader queued behind the build would still be blocked here
        alive = t.is_alive()
        build.release.set()
        t.join(10)
    assert not alive, "a recall with a cached graph waited on another scope's build"
    assert isinstance(got["r"], dict)


def test_a_scope_being_built_reports_warming_instead_of_waiting(tmp_path, monkeypatch):
    channel = _channel(tmp_path)
    with _BlockedBuild(channel, "a", monkeypatch):
        assert adjacency_rcu.build_in_flight(channel, ("a", False, False))
        got = egw.score_candidates_unless_warming(channel, QUERY, ["f_a"], "a")
    assert got is None
    # Built and swapped in: the next recall gets the boost back.
    assert ("a", False, False) in channel._adj_slots
    assert isinstance(egw.score_candidates_unless_warming(channel, QUERY, ["f_a"], "a"), dict)
    assert not adjacency_rcu.build_in_flight(channel, ("a", False, False))


def test_with_no_build_in_flight_a_recall_builds_and_scores_as_before(tmp_path):
    channel = _channel(tmp_path)
    assert isinstance(egw.score_candidates_unless_warming(channel, QUERY, ["f_a"], "a"), dict)
    assert ("a", False, False) in channel._adj_slots


def test_the_warmup_builds_the_graph(tmp_path):
    channel = _channel(tmp_path)
    engine = SimpleNamespace(_retrieval_engine=SimpleNamespace(_entity=channel))
    assert egw.warm(engine, "a") is True
    assert ("a", False, False) in channel._adj_slots


class _Runtime:
    class _Lease:
        def __enter__(self):
            return object()

        def __exit__(self, *exc):
            return False

    def operation_nowait(self):
        return self._Lease()


def test_warmup_builds_the_graph_before_its_first_recall_and_reports_progress(
        tmp_path, monkeypatch):
    channel = _channel(tmp_path)
    order: list[str] = []
    phases: list[dict] = []

    def recall(query, limit=5, fast=True):
        order.append("graph" if ("a", False, False) in channel._adj_slots else "cold")
        phases.append(recall_warmup.warmup_status())

    engine = SimpleNamespace(profile_id="a", recall=recall,
                             _retrieval_engine=SimpleNamespace(_entity=channel))
    monkeypatch.setattr(recall_warmup, "_state", {"phase": "pending"})
    assert recall_warmup.warmup_status()["warm"] is False
    recall_warmup.run_warmup_recalls(engine, _Runtime(),
                                     warm_spreading_activation=lambda e, r: None)
    assert order and set(order) == {"graph"}
    assert all(p["phase"] == "recalls" and p["warming"] for p in phases)
    assert recall_warmup.warmup_status() == {"phase": "warm", "warm": True, "warming": []}


@pytest.mark.parametrize("building", [True, False])
def test_recall_reports_a_skipped_entity_graph_as_incomplete(
        engine_with_mock_deps, monkeypatch, building):
    """RetrievalEngine.recall, composed: the skip is in incomplete_channels."""
    from tests.conftest import force_sync_enrichment

    engine = force_sync_enrichment(engine_with_mock_deps)
    engine.store("Alice Johnson moved the build cache to the Frankfurt runner.")
    channel = engine._retrieval_engine._entity
    channel.invalidate_cache()
    if building:
        with _BlockedBuild(channel, engine.profile_id, monkeypatch):
            response = engine._retrieval_engine.recall(
                "Where did Alice Johnson move the build cache?", engine.profile_id)
        assert response.channel_status["entity_graph"] == chstat.WARMING
        assert "entity_graph" in response.incomplete_channels
    else:
        response = engine._retrieval_engine.recall(
            "Where did Alice Johnson move the build cache?", engine.profile_id)
        assert response.channel_status["entity_graph"] != chstat.WARMING
        assert "entity_graph" not in response.incomplete_channels


def _wait_until(predicate, seconds: float = 10.0) -> bool:
    done = threading.Event()
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return True
        done.wait(0.01)
    return predicate()


def test_in_the_daemon_a_cold_graph_is_built_beside_the_recall(tmp_path, monkeypatch):
    """The daemon's policy: no graph build ever runs on a recall's clock."""
    channel = _channel(tmp_path)
    monkeypatch.setattr(adjacency_rcu, "_background", True)
    assert egw.score_candidates_unless_warming(channel, QUERY, ["f_a"], "a") is None
    assert _wait_until(lambda: ("a", False, False) in channel._adj_slots
                       and not adjacency_rcu.build_in_flight(channel, ("a", False, False)))
    assert isinstance(egw.score_candidates_unless_warming(channel, QUERY, ["f_a"], "a"), dict)


def test_begin_starts_the_graph_before_the_model_and_the_warmup_waits_for_it(
        tmp_path, monkeypatch):
    channel = _channel(tmp_path)
    monkeypatch.setattr(adjacency_rcu, "_background", False)
    monkeypatch.setattr(recall_warmup, "_state", {"phase": "pending"})
    seen: list[bool] = []
    engine = SimpleNamespace(
        profile_id="a", _retrieval_engine=SimpleNamespace(_entity=channel),
        recall=lambda q, limit=5, fast=True: seen.append(("a", False, False)
                                                          in channel._adj_slots))
    thread = recall_warmup.begin(engine)
    assert thread is not None and adjacency_rcu._background is True
    recall_warmup.run_warmup_recalls(engine, _Runtime(),
                                     warm_spreading_activation=lambda e, r: None)
    assert seen and all(seen)
