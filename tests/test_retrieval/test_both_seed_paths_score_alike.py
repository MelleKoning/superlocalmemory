# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The graph walk's seeds score the same whichever search found them (MUSE-4).

The vector-index path scored a seed ``max(0, cos)``; the SQL fallback, used when
the index is unavailable, scored the same memory ``(cos + 1) / 2`` — cos 0.45
became 0.45 on one path and 0.725 on the other, and an unrelated memory (cos 0)
started at 0.5. The walk's answer therefore depended on which path ran, and the
cache key did not say which, so one path's cached walk answered the other's.
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.spreading_activation import (
    SpreadingActivation,
    SpreadingActivationConfig,
    _seed_score,
)
from tests.test_retrieval.cross_scope_fixture import REQ, PartitionedVS, build_store


@pytest.fixture(autouse=True)
def _no_background_writes(monkeypatch):
    import superlocalmemory.storage.deferred_writes as dw
    monkeypatch.setattr(dw, "submit_background", lambda fn: None)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("muse4") / "sa.db")


def _paths(store):
    cfg = SpreadingActivationConfig()
    vec = SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("L")), cfg)
    sql = SpreadingActivation(store.db, None, cfg)
    return vec, sql


def test_one_scale() -> None:
    assert _seed_score(0.45, True) == _seed_score(0.45, False) == 0.45
    assert _seed_score(-0.3, True) == _seed_score(-0.3, False) == 0.0


def test_both_paths_give_identical_seed_scores(store) -> None:
    vec, sql = _paths(store)
    for q in store.queries:
        kw = dict(include_global=False, include_shared=False)
        assert vec._seed_search(q.tolist(), REQ, **kw) == sql._seed_search(
            q.tolist(), REQ, **kw)


def test_both_paths_give_the_same_answer(store) -> None:
    vec, sql = _paths(store)
    for q in store.queries:
        assert vec.search(q.tolist(), REQ, top_k=10) == sql.search(q.tolist(), REQ, top_k=10)


def test_the_cache_key_names_the_path(store) -> None:
    vec, sql = _paths(store)
    q = store.queries[0].tolist()
    seeds = vec._seed_search(q, REQ, include_global=False, include_shared=False)
    assert (vec._compute_query_hash(q, REQ, seeds=seeds)
            != sql._compute_query_hash(q, REQ, seeds=seeds))
