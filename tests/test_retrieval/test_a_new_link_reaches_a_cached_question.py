# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A link added, re-weighted or removed after a question was asked is walked the next time.

The activation cache was keyed by the question, the scope and the seed set.
The walk also depends on the links it follows, so a change to the links that
left the seeds alone stayed hidden behind the cached answer for up to an hour.
The key now carries the graph's generation (``storage/graph_generation.py``).

Every test compares the repeat answer with a FRESH walk of the changed store
(cache emptied first) and checks the change really altered that answer, so a
pass cannot come from the change having had no effect.
"""

from __future__ import annotations

import pytest

import superlocalmemory.storage.deferred_writes as dw
from superlocalmemory.retrieval.spreading_activation import (
    SpreadingActivation,
    SpreadingActivationConfig,
)
from tests.test_retrieval.cross_scope_fixture import OWN, REQ, PartitionedVS, build_store


@pytest.fixture()
def store(tmp_path):
    return build_store(tmp_path / "sa.db")


def _channel(store) -> SpreadingActivation:
    return SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("LGS")),
                               SpreadingActivationConfig())


def _search(ch, q, *, cross: bool):
    out = ch.search(q, REQ, top_k=10, include_global=cross, include_shared=cross)
    dw._bg_queue.join()
    return out


def _fresh(store, q, *, cross: bool):
    """The answer a walk of the store as it is now gives, with no cache."""
    store.db.execute("DELETE FROM activation_cache")
    return _search(_channel(store), q, cross=cross)


def _seeds(ch, q, *, cross: bool):
    seeds = ch._seed_search(q, REQ, include_global=cross, include_shared=cross)
    if cross:
        seeds = ch._merge_cross_scope_seeds(
            q, seeds, REQ, include_global=True, include_shared=True)
    return sorted(seeds)


def _cached_rows(store) -> int:
    return store.db.execute("SELECT COUNT(*) AS n FROM activation_cache")[0]["n"]


def _link_seeds_to(store, seeds, target, owner, scope, table="graph_edges"):
    for i, (seed, _score) in enumerate(seeds):
        if table == "graph_edges":
            store.db.execute(
                "INSERT INTO graph_edges (edge_id, profile_id, scope, source_id,"
                " target_id, edge_type, weight) VALUES (?,?,?,?,?,'semantic',1.0)",
                (f"new{i}", owner, scope, seed, target))
        else:
            store.db.execute(
                "INSERT INTO association_edges (edge_id, profile_id, source_fact_id,"
                " target_fact_id, association_type, weight) VALUES"
                " (?,?,?,?,'auto_link',1.0)", (f"assoc{i}", owner, seed, target))


def _unseen_target(store, first, seeds, prefix):
    taken = {f for f, _ in first} | {f for f, _ in seeds}
    return next(f for f in store.ids(prefix) if f not in taken)


def _assert_reflected(store, ch, q, first, seeds_before, *, cross: bool):
    assert _seeds(ch, q, cross=cross) == seeds_before, "the change moved the seeds"
    again = _search(ch, q, cross=cross)
    fresh = _fresh(store, q, cross=cross)
    assert fresh != first, "the change did not alter the walk; the test proves nothing"
    assert again == fresh
    return again


class TestALinkChangeIsWalked:
    def test_a_new_link_reaches_the_repeat_question(self, store) -> None:
        q = store.queries[0].tolist()
        ch = _channel(store)
        first = _search(ch, q, cross=False)
        assert first and _cached_rows(store) > 0
        seeds = _seeds(ch, q, cross=False)
        target = _unseen_target(store, first, seeds, "L")
        _link_seeds_to(store, seeds, target, REQ, "personal")
        again = _assert_reflected(store, ch, q, first, seeds, cross=False)
        assert target in {f for f, _ in again}

    def test_a_new_association_link_reaches_the_repeat_question(self, store) -> None:
        q = store.queries[1].tolist()
        ch = _channel(store)
        first = _search(ch, q, cross=False)
        seeds = _seeds(ch, q, cross=False)
        target = _unseen_target(store, first, seeds, "L")
        _link_seeds_to(store, seeds, target, REQ, None, table="association_edges")
        _assert_reflected(store, ch, q, first, seeds, cross=False)

    def test_a_reweighted_link_reaches_the_repeat_question(self, store) -> None:
        q = store.queries[2].tolist()
        ch = _channel(store)
        first = _search(ch, q, cross=False)
        seeds = _seeds(ch, q, cross=False)
        for seed, _score in seeds:
            store.db.execute(
                "UPDATE graph_edges SET weight = 1.0 - weight"
                " WHERE source_id = ? OR target_id = ?", (seed, seed))
        _assert_reflected(store, ch, q, first, seeds, cross=False)

    def test_a_removed_link_reaches_the_repeat_question(self, store) -> None:
        q = store.queries[3].tolist()
        ch = _channel(store)
        first = _search(ch, q, cross=False)
        seeds = _seeds(ch, q, cross=False)
        for seed, _score in seeds:
            store.db.execute(
                "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
                (seed, seed))
        _assert_reflected(store, ch, q, first, seeds, cross=False)

    def test_another_profiles_global_link_reaches_a_cross_scope_question(self, store) -> None:
        q = store.queries[4].tolist()
        ch = _channel(store)
        first = _search(ch, q, cross=True)
        seeds = _seeds(ch, q, cross=True)
        target = _unseen_target(store, first, seeds, "G")
        # The owner writes the link: the requester's own counter does not move.
        _link_seeds_to(store, seeds, target, OWN, "global")
        again = _assert_reflected(store, ch, q, first, seeds, cross=True)
        assert target in {f for f, _ in again}


class TestTheCacheStillServesAnUnchangedGraph:
    def _count_propagations(self, monkeypatch) -> dict:
        calls = {"n": 0}
        real = SpreadingActivation._propagate

        def counting(self_, *a, **k):
            calls["n"] += 1
            return real(self_, *a, **k)

        monkeypatch.setattr(SpreadingActivation, "_propagate", counting)
        return calls

    def test_another_profiles_private_link_keeps_a_personal_entry(
        self, store, monkeypatch,
    ) -> None:
        calls = self._count_propagations(monkeypatch)
        q = store.queries[5].tolist()
        ch = _channel(store)
        first = _search(ch, q, cross=False)
        store.db.execute(
            "INSERT INTO graph_edges (edge_id, profile_id, scope, source_id,"
            " target_id, edge_type, weight) VALUES ('other', ?, 'personal',"
            " 'P00000', 'P00001', 'semantic', 1.0)", (OWN,))
        assert _search(ch, q, cross=False) == first
        assert calls["n"] == 1

    def test_a_store_that_cannot_say_its_generation_does_not_use_the_cache(
        self, store, monkeypatch,
    ) -> None:
        calls = self._count_propagations(monkeypatch)
        with store.db.raw_connection() as conn:
            conn.execute("DROP TABLE graph_generation")
        q = store.queries[6].tolist()
        ch = _channel(store)
        first = _search(ch, q, cross=False)
        assert first
        assert _search(ch, q, cross=False) == first
        assert calls["n"] == 2, "an entry was served without proof the graph is unchanged"
        assert _cached_rows(store) == 0


class TestDeletingAProfileEmptiesTheCache:
    def test_profile_delete_drops_every_entry(self, store) -> None:
        q = store.queries[7].tolist()
        _search(_channel(store), q, cross=True)
        assert _cached_rows(store) > 0
        store.db.execute("DELETE FROM profiles WHERE profile_id = ?", (OWN,))
        assert _cached_rows(store) == 0
        rows = store.db.execute(
            "SELECT profile_id FROM graph_generation WHERE profile_id = ?", (OWN,))
        assert rows == []
