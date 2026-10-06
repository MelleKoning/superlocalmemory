# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A kind-filtered recall looks for that kind, not only filters for it afterwards.

Two defects on the kind path, both measured on a copy of a 21,739-fact store:

* the facet filter re-ran its database read once per candidate id, so one
  kind-filtered recall decoded about 170,000 kind rows (3-20 s of the recall);
* the kind filter only ran AFTER fusion, so a memory of that kind ranked below
  the unfiltered pool was never a candidate, and the answer came back empty
  although the memory was stored (3 of 10 everyday decision questions).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from superlocalmemory.retrieval import facets as facets_mod
from superlocalmemory.retrieval import kind_scope
from superlocalmemory.retrieval.bm25_channel import BM25Channel
from superlocalmemory.retrieval.canonical_vector_index import CanonicalVectorIndex
from superlocalmemory.retrieval.facets import Facets, matching_fact_ids
from superlocalmemory.retrieval.semantic_channel import SemanticChannel
from superlocalmemory.storage.memory_kinds import kind_fields
from tests.test_retrieval.cross_scope_fixture import DIM, REQ, build_store


@pytest.fixture()
def store(tmp_path):
    return build_store(tmp_path / "k.db")


def _set_kind(store, fid: str, kind: str, source: str = "caller", conf=None) -> None:
    store.db.execute(
        "UPDATE atomic_facts SET memory_kind = ?, memory_kind_source = ?, "
        "memory_kind_confidence = ? WHERE fact_id = ?", (kind, source, conf, fid))


def _engine(store) -> SimpleNamespace:
    index = CanonicalVectorIndex(store.db, DIM)
    eng = SimpleNamespace(
        _semantic=SemanticChannel(store.db, vector_store=index), _bm25=BM25Channel(store.db),
        _kind_vectors=index, _kind_membership=kind_scope.KindMembership(store.db),
        _config=SimpleNamespace(semantic_top_k=10, bm25_top_k=10),
        _display_min_confidence=0.20,
    )
    return eng


def test_each_facet_filter_reads_the_store_once_per_recall(store, monkeypatch) -> None:
    calls = {"n": 0}
    real = facets_mod._by_kind

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(facets_mod, "_by_kind", counting)
    ids = store.ids("L")
    for fid in ids[:7]:
        _set_kind(store, fid, "decision")
    kept = matching_fact_ids(store.db, ids, REQ, Facets.of(kind="decision"))
    assert kept == set(ids[:7])
    assert calls["n"] == 1


def test_the_kind_index_agrees_with_the_one_kind_serializer(store) -> None:
    ids = store.ids("L")
    _set_kind(store, ids[0], "decision")                       # confirmed
    _set_kind(store, ids[1], "decision", "model:laya", 0.9)    # suggested, shown
    _set_kind(store, ids[2], "decision", "model:laya", 0.05)   # below threshold
    _set_kind(store, ids[3], "Choice")                         # an alias, mixed case
    membership = kind_scope.KindMembership(store.db)
    got = membership.ids_of(REQ, "decision", 0.20)
    rows = store.db.execute(
        "SELECT fact_id, memory_kind, memory_kind_source, memory_kind_confidence, "
        "fact_type FROM atomic_facts WHERE profile_id = ?", (REQ,))
    want = {dict(r)["fact_id"] for r in rows
            if kind_fields(dict(r), display_min_confidence=0.20)["memory_kind"] == "decision"}
    assert got == want == {ids[0], ids[1], ids[3]}


def test_a_kind_changed_by_any_writer_counts_from_the_next_recall(store) -> None:
    membership = kind_scope.KindMembership(store.db)
    fid = store.ids("L")[5]
    assert fid not in membership.ids_of(REQ, "rule", 0.20)
    _set_kind(store, fid, "rule")
    assert fid in membership.ids_of(REQ, "rule", 0.20)
    store.db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (fid,))
    assert fid not in membership.ids_of(REQ, "rule", 0.20)


def test_the_superset_clause_never_drops_a_member(store) -> None:
    ids = store.ids("L")
    _set_kind(store, ids[0], "prospective")
    store.db.execute("UPDATE atomic_facts SET fact_type = 'prospective' WHERE fact_id = ?",
                     (ids[1],))  # never classified: shown by its legacy type
    clause, params = kind_scope.kind_superset_clause("prospective", prefix="")
    rows = store.db.execute(f"SELECT fact_id FROM atomic_facts WHERE 1=1{clause}",
                            tuple(params))
    found = {dict(r)["fact_id"] for r in rows}
    members = kind_scope.KindMembership(store.db).ids_of(REQ, "prospective", 0.20)
    assert {ids[0], ids[1]} <= members <= found


def test_a_memory_of_the_kind_outside_the_ordinary_pool_becomes_a_candidate(store) -> None:
    eng = _engine(store)
    q = store.queries[0]
    ordinary = eng._semantic.search(q.tolist(), REQ, top_k=10)
    pool = {f for f, _ in ordinary}
    outside = next(f for f in reversed(store.ids("L")) if f not in pool)
    _set_kind(store, outside, "decision")
    got = kind_scope.supplement(
        eng, {"semantic": ordinary}, query="unrelated words", query_embedding=q.tolist(),
        profile_id=REQ, kind="decision")
    assert outside in {f for f, _ in got["semantic"]}
    # The ordinary hits keep their own scores; the list stays ranked.
    merged = dict(got["semantic"])
    assert all(np.isclose(merged[f], s) for f, s in ordinary)
    assert got["semantic"] == sorted(got["semantic"], key=lambda x: (-x[1], x[0]))


def test_exact_words_of_the_kind_are_found_by_the_in_kind_word_search(store) -> None:
    eng = _engine(store)
    target = store.ids("L")[40]
    _set_kind(store, target, "procedure")
    got = kind_scope.supplement(
        eng, {}, query=f"content {target}", query_embedding=None,
        profile_id=REQ, kind="procedure")
    assert [f for f, _ in got.get("bm25", [])] == [target]


def test_a_supplement_that_cannot_run_leaves_the_answer_unchanged(store) -> None:
    eng = _engine(store)
    eng._kind_membership = SimpleNamespace(ids_of=lambda *a: (_ for _ in ()).throw(RuntimeError()))
    base = {"semantic": [("x", 0.9)]}
    assert kind_scope.supplement(eng, base, query="q", query_embedding=None,
                                 profile_id=REQ, kind="decision") == base
