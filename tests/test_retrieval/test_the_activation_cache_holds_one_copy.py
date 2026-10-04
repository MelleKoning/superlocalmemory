# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A cached graph walk is stored, and answered, once (R2).

Two identical recalls in flight at once each stored the entry, with a fresh
row id per row, so the table held every node twice and the next hit returned
every node twice — five distinct memories filling ten slots.
"""

from __future__ import annotations

import threading

import pytest

import superlocalmemory.storage.deferred_writes as dw
from superlocalmemory.retrieval.spreading_activation import (
    SpreadingActivation,
    SpreadingActivationConfig,
)
from tests.test_retrieval.cross_scope_fixture import REQ, PartitionedVS, build_store


@pytest.fixture()
def store(tmp_path):
    return build_store(tmp_path / "sa.db")


def _channel(store) -> SpreadingActivation:
    return SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("L")),
                               SpreadingActivationConfig())


def _duplicated_nodes(db) -> int:
    return len(db.execute(
        "SELECT profile_id, query_hash, node_id, COUNT(*) AS c FROM activation_cache "
        "GROUP BY profile_id, query_hash, node_id HAVING c > 1", ()))


class TestOneCopy:
    def test_two_writers_of_one_entry_leave_one_copy(self, store) -> None:
        ch = _channel(store)
        acts = {"L00001": 0.9, "L00002": 0.5}
        ch._cache_results("k", REQ, acts)
        ch._cache_results("k", REQ, {"L00001": 0.8, "L00003": 0.4})
        rows = store.db.execute(
            "SELECT node_id, activation_value FROM activation_cache WHERE query_hash='k'"
            " ORDER BY node_id", ())
        # The second write replaced the entry; nothing of the first lingers.
        assert [(r["node_id"], r["activation_value"]) for r in rows] == [
            ("L00001", 0.8), ("L00003", 0.4)]

    def test_concurrent_identical_recalls_store_one_copy(self, store) -> None:
        ch = _channel(store)
        q = store.queries[0].tolist()
        gate = threading.Barrier(2)
        outs: list = []

        def go() -> None:
            gate.wait()
            outs.append(ch.search(q, REQ, top_k=10))

        threads = [threading.Thread(target=go) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dw._bg_queue.join()
        assert _duplicated_nodes(store.db) == 0
        hit = ch.search(q, REQ, top_k=10)
        assert hit == outs[0]
        assert len({f for f, _ in hit}) == len(hit)

    def test_a_store_already_holding_duplicates_answers_each_node_once(self, store) -> None:
        """Stores written before the fix still hold doubled entries for an hour."""
        ch = _channel(store)
        q = store.queries[0].tolist()
        fresh = ch.search(q, REQ, top_k=10)
        dw._bg_queue.join()
        store.db.execute(
            "INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id,"
            " activation_value, iteration, created_at, expires_at)"
            " SELECT cache_id || '-dup', profile_id, query_hash, node_id,"
            " activation_value, iteration, created_at, expires_at FROM activation_cache",
            ())
        assert _duplicated_nodes(store.db) > 0
        assert ch.search(q, REQ, top_k=10) == fresh
