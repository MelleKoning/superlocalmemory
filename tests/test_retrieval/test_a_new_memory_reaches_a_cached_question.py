# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A memory saved after a question was asked is walked the next time (R3).

The entry was keyed by the question and scope only. A memory saved after
the question was asked — even the best match, now the first seed — stayed
invisible to that question until the entry expired an hour later.
"""

from __future__ import annotations

import numpy as np
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



class TestANewMemoryIsWalked:
    def test_a_new_best_match_reaches_the_repeat_question(self, store) -> None:
        q = store.queries[0]
        first = _channel(store).search(q.tolist(), REQ, top_k=10)
        dw._bg_queue.join()
        assert first

        new_id = "LNEW0"
        store.embs[new_id] = q.astype(np.float32)
        with store.db.raw_connection() as conn:
            conn.execute("INSERT INTO memories (memory_id, profile_id, scope, content)"
                         " VALUES ('m_new', ?, 'personal', 'new')", (REQ,))
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, scope, content,"
                " fact_type, confidence, importance, evidence_count, access_count,"
                " embedding, created_at) VALUES (?, 'm_new', ?, 'personal', 'new exact"
                " match', 'semantic', 0.9, 0.5, 1, 0, ?, datetime('now'))",
                (new_id, REQ, q.astype(np.float32).tobytes()))
        ch = _channel(store)
        assert ch._seed_search(q.tolist(), REQ, include_global=False,
                               include_shared=False)[0][0] == new_id
        again = ch.search(q.tolist(), REQ, top_k=10)
        assert new_id in {f for f, _ in again}

    def test_an_unchanged_store_still_reuses_the_entry(self, store, monkeypatch) -> None:
        calls = {"n": 0}
        real = SpreadingActivation._propagate

        def counting(self_, *a, **k):
            calls["n"] += 1
            return real(self_, *a, **k)

        monkeypatch.setattr(SpreadingActivation, "_propagate", counting)
        ch = _channel(store)
        q = store.queries[2].tolist()
        first = ch.search(q, REQ, top_k=10)
        dw._bg_queue.join()
        assert ch.search(q, REQ, top_k=10) == first
        assert calls["n"] == 1

    def test_the_key_ignores_seed_order_and_float_noise(self, store) -> None:
        ch = _channel(store)
        a = ch._compute_query_hash([0.1] * 4, REQ, seeds=[("x", 0.5), ("y", 0.25)])
        b = ch._compute_query_hash([0.1] * 4, REQ,
                                   seeds=[("y", 0.25 + 1e-12), ("x", 0.5)])
        c = ch._compute_query_hash([0.1] * 4, REQ, seeds=[("x", 0.5), ("z", 0.25)])
        assert a == b and a != c
