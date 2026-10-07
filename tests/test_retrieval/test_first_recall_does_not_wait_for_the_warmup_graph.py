# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""The first recall after start does not queue behind the warm-up's graph build.

Measured on a copy of a 21,738-fact store: the first recall after the daemon
reported ready took 8.5-13 s while its channels took 1.2 s; ~6 s was spent
waiting on the entity graph's cache lock held by the start-up warm-up. Now:

* the warm-up builds the entity graph directly, before its own recalls;
* while the warm-up holds the graph, a recall reports ``entity_graph`` as
  ``warming`` (incomplete) instead of waiting, and gets the boost back once
  the graph is built;
* outside the warm-up, a recall that finds no graph builds it itself, as
  before (a one-shot command has no warm-up);
* /health says what the warm-up is still building.

Real ``EntityGraphChannel`` over a real SQLite store; waits are event-based.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from superlocalmemory.retrieval import entity_graph_warmup as egw
from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
from superlocalmemory.server import recall_warmup
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager

PID = "default"


def _channel(tmp_path) -> EntityGraphChannel:
    db = DatabaseManager(tmp_path / "warm_graph.db")
    with db.raw_connection() as conn:
        schema.create_all_tables(conn)
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                     (PID, PID))
        for i in range(2):
            conn.execute(
                "INSERT INTO memories (memory_id, profile_id, scope, shared_with, content) "
                "VALUES (?, ?, 'personal', NULL, ?)", (f"m{i}", PID, f"note {i}"))
            conn.execute(
                "INSERT INTO atomic_facts "
                "(fact_id, memory_id, profile_id, scope, shared_with, content, "
                " fact_type, confidence, importance, evidence_count, access_count, "
                " canonical_entities_json, embedding, created_at) "
                "VALUES (?, ?, ?, 'personal', NULL, ?, 'semantic', 0.9, 0.5, 1, 0, "
                "'[]', ?, datetime('now'))",
                (f"f{i}", f"m{i}", PID, f"note {i}", json.dumps([1.0, 0.0, 0.0, 0.0])))
        conn.commit()
    return EntityGraphChannel(db)


def _held_by_another_thread(lock):
    """Hold ``lock`` on another thread until the returned event is set."""
    held, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            held.set()
            release.wait(10)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert held.wait(5)
    return release, t


def test_a_recall_does_not_wait_while_the_warmup_holds_the_graph(tmp_path):
    channel = _channel(tmp_path)
    with egw.building(channel):
        release, t = _held_by_another_thread(channel._cache_lock)
        try:
            # Would block until ``release`` if it queued behind the build.
            got = egw.score_candidates_unless_warming(channel, "Alice met Bob", ["f0", "f1"], PID)
        finally:
            release.set()
            t.join(5)
    assert got is None


def test_outside_the_warmup_a_recall_builds_and_scores_as_before(tmp_path):
    channel = _channel(tmp_path)
    got = egw.score_candidates_unless_warming(channel, "Alice met Bob", ["f0", "f1"], PID)
    assert isinstance(got, dict)


def test_the_warmup_builds_the_graph_and_a_later_recall_scores(tmp_path):
    channel = _channel(tmp_path)
    engine = SimpleNamespace(_retrieval_engine=SimpleNamespace(_entity=channel))
    assert egw.warm(engine, PID) is True
    assert (PID, False, False) in channel._adj_slots
    with egw.building(channel):  # graph built, lock free: no reason to skip it
        assert isinstance(egw.score_candidates_unless_warming(channel, "Alice met Bob", ["f0"], PID), dict)
    assert not egw.is_building(channel)


def test_building_scopes_nest(tmp_path):
    channel = _channel(tmp_path)
    with egw.building(channel):
        with egw.building(channel):
            pass
        assert egw.is_building(channel)  # the outer warm-up is still running
    assert not egw.is_building(channel)


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
        order.append("graph" if (PID, False, False) in channel._adj_slots else "cold")
        phases.append(recall_warmup.warmup_status())

    engine = SimpleNamespace(profile_id=PID, recall=recall,
                             _retrieval_engine=SimpleNamespace(_entity=channel))
    monkeypatch.setattr(recall_warmup, "_state", {"phase": "pending"})
    assert recall_warmup.warmup_status()["warm"] is False
    recall_warmup.run_warmup_recalls(engine, _Runtime(),
                                     warm_spreading_activation=lambda e, r: None)
    assert order and set(order) == {"graph"}
    assert all(p["phase"] == "recalls" and p["warming"] for p in phases)
    assert recall_warmup.warmup_status() == {"phase": "warm", "warm": True, "warming": []}
