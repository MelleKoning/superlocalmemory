# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Narrowing and boosting by project, on the store directly (GitHub #150)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from superlocalmemory.retrieval.facets import Facets, list_facets, matching_fact_ids
from superlocalmemory.retrieval.project_scope import (
    BOOST,
    boost_order,
    lift,
    narrow,
    stored_projects,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _save(db, content, *, project=None, agent=None) -> str:
    meta = {k: v for k, v in (("project", project), ("agent_id", agent)) if v is not None}
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content, metadata=meta))
    return db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                                    fact_type=FactType.SEMANTIC))


@dataclass(frozen=True)
class _Fused:
    fact_id: str
    fused_score: float


def test_stored_projects_reads_what_each_memory_was_saved_under(db) -> None:
    a = _save(db, "a", project="/Users/x/slm")
    b = _save(db, "b", project="  ")
    c = _save(db, "c")
    assert stored_projects(db, [a, b, c]) == {a: "/Users/x/slm"}


def test_malformed_metadata_is_skipped_not_fatal(db) -> None:
    a = _save(db, "a", project="slm")
    bad = _save(db, "bad", project="slm")
    db.execute("UPDATE memories SET metadata_json = '{not json' WHERE memory_id = "
               "(SELECT memory_id FROM atomic_facts WHERE fact_id = ?)", (bad,))
    assert stored_projects(db, [a, bad]) == {a: "slm"}


def test_filter_by_name_matches_a_saved_path(db) -> None:
    path = _save(db, "a", project="/Users/x/SLM/")
    name = _save(db, "b", project="slm")
    other = _save(db, "c", project="/Users/x/other")
    legacy = _save(db, "d")
    scoped = narrow(db, [path, name, other, legacy], "default", Facets.of(project="SLM"))
    assert scoped.kept == (path, name)
    assert scoped.report == {"filter": {
        "project": "SLM", "key": "slm", "applied": True, "strict": False, "matched": 2,
        "note": "", "identity": {"name": "SLM", "rule": "last folder name, ignoring case"}}}


def test_filter_matching_nothing_keeps_everything_and_says_why(db) -> None:
    ids = [_save(db, "a", project="slm"), _save(db, "b")]
    scoped = narrow(db, ids, "default", Facets.of(project="ghost"))
    assert scoped.kept == tuple(ids)
    report = scoped.report["filter"]
    assert report["applied"] is False and report["reason"] == "no_match"
    assert "ghost" in report["note"]


def test_an_unreadable_store_falls_back_and_says_so(db) -> None:
    ids = [_save(db, "a", project="slm")]

    class _Broken:
        def execute(self, *a, **k):
            raise RuntimeError("disk gone")

    scoped = narrow(_Broken(), ids, "default", Facets.of(project="slm"))
    assert scoped.kept == tuple(ids)
    assert scoped.report["filter"]["reason"] == "unreadable"


def test_other_facets_stay_hard_filters(db) -> None:
    a = _save(db, "a", project="slm", agent="claude")
    b = _save(db, "b", project="slm", agent="grok")
    assert narrow(db, [a, b], "default", Facets.of(project="slm", agent="grok")).kept == (b,)
    # The project fall-back never loosens another facet: nothing saved by
    # "codex" stays nothing.
    assert narrow(db, [a, b], "default", Facets.of(project="ghost", agent="codex")).kept == ()


def test_prefer_marks_without_removing(db) -> None:
    a = _save(db, "a", project="/x/slm")
    b = _save(db, "b", project="other")
    c = _save(db, "c")
    scoped = narrow(db, [a, b, c], "default", Facets.of(prefer_project="SLM"))
    assert scoped.kept == (a, b, c)
    assert scoped.preferred == frozenset({a})
    assert scoped.report["prefer"]["matched"] == 1


def test_no_project_means_no_report(db) -> None:
    a = _save(db, "a", agent="claude")
    assert narrow(db, [a], "default", Facets.of(agent="claude")).report is None


def test_matching_fact_ids_project_uses_the_same_rule(db) -> None:
    path = _save(db, "a", project="/Users/x/SuperLocalMemory")
    assert matching_fact_ids(db, [path], "default", Facets.of(project="superlocalmemory")) == {path}
    # prefer_project alone narrows nothing.
    assert matching_fact_ids(db, [path], "default", Facets.of(prefer_project="zzz")) == {path}


def test_list_facets_groups_a_project_saved_as_name_and_path(db) -> None:
    _save(db, "1", project="slm")
    _save(db, "2", project="/Users/x/SLM/")
    _save(db, "3", project="loops")
    assert list_facets(db, "default")["projects"] == [
        {"name": "slm", "memories": 2}, {"name": "loops", "memories": 1}]


def test_facets_empty_and_narrows() -> None:
    assert Facets.of().empty and not Facets.of().narrows
    prefer = Facets.of(prefer_project="slm")
    assert not prefer.empty and not prefer.narrows
    assert Facets.of(project="slm").narrows


# -- the boost ---------------------------------------------------------------


def test_lift_is_bounded_and_never_sinks_a_negative_score() -> None:
    assert lift(1.0) == pytest.approx(1.0 + BOOST)
    assert lift(-1.0) == pytest.approx(-1.0 + BOOST)
    assert lift(0.0) == 0.0


def test_boost_passes_only_a_close_neighbour() -> None:
    items = [_Fused("strong", 1.0), _Fused("close", 0.85), _Fused("weak", 0.5)]
    out = boost_order(items, frozenset({"close", "weak"}))
    assert [i.fact_id for i in out] == ["close", "strong", "weak"]


def test_boost_exactly_at_the_bound_does_not_pass() -> None:
    items = [_Fused("strong", 1.0), _Fused("edge", 1 / (1 + BOOST))]
    assert [i.fact_id for i in boost_order(items, frozenset({"edge"}))] == ["strong", "edge"]


def test_boost_keeps_incoming_order_on_ties_and_adds_nothing() -> None:
    items = [_Fused("a", 0.5), _Fused("b", 0.5), _Fused("c", 0.5)]
    out = boost_order(items, frozenset({"b", "c"}))
    assert [i.fact_id for i in out] == ["b", "c", "a"]
    assert boost_order(items, frozenset()) == items
    assert sorted(i.fact_id for i in out) == ["a", "b", "c"]


def test_a_long_path_keeps_its_project(db) -> None:
    path = "/" + "/".join(["deep"] * 80) + "/acme-billing"
    assert len(path) > 200
    a = _save(db, "a", project="acme-billing")
    assert narrow(db, [a], "default", Facets.of(project=path)).kept == (a,)
    assert Facets.of(prefer_project=path).prefer_project == path


# -- the final order (after learned ranking) ----------------------------------


def _result(fid: str, key: float):
    from superlocalmemory.storage.models import RetrievalResult

    return RetrievalResult(fact=AtomicFact(fact_id=fid, content=fid), score=0.5,
                           ranking_score=key, evidence_chain=["bm25(rank=1)"])


def test_final_order_passes_only_close_neighbours_and_keeps_the_rest() -> None:
    from superlocalmemory.retrieval.project_scope import prefer_in_final_order

    results = [_result("t1", 1.0), _result("t2", 0.9), _result("p1", 0.85),
               _result("t3", 0.5), _result("p2", 0.3)]
    out = prefer_in_final_order(results, frozenset({"p1", "p2"}))
    assert [r.fact.fact_id for r in out] == ["p1", "t1", "t2", "t3", "p2"]
    assert [r.rank_position for r in out] == [1, 2, 3, 4, 5]
    lifted = {r.fact.fact_id: r for r in out}
    assert lifted["p1"].ranking_score == pytest.approx(0.85 * 1.25)
    assert lifted["p2"].evidence_chain[-1] == "same_project"
    assert lifted["t1"].ranking_score == 1.0 and "same_project" not in lifted["t1"].evidence_chain
    assert results[2].ranking_score == 0.85 and results[2].evidence_chain == ["bm25(rank=1)"]


def test_final_order_never_reorders_two_project_results() -> None:
    from superlocalmemory.retrieval.project_scope import prefer_in_final_order

    results = [_result("p1", 0.5), _result("p2", 0.6)]  # learned order, kept
    out = prefer_in_final_order(results, frozenset({"p1", "p2"}))
    assert [r.fact.fact_id for r in out] == ["p1", "p2"]


def test_final_order_uses_score_when_no_ranking_score() -> None:
    from superlocalmemory.retrieval.project_scope import prefer_in_final_order
    from superlocalmemory.storage.models import RetrievalResult

    a = RetrievalResult(fact=AtomicFact(fact_id="t", content="t"), score=0.6)
    b = RetrievalResult(fact=AtomicFact(fact_id="p", content="p"), score=0.5)
    assert [r.fact.fact_id for r in prefer_in_final_order([a, b], frozenset({"p"}))] == ["p", "t"]


def test_final_order_without_matches_is_the_same_list() -> None:
    from superlocalmemory.retrieval.project_scope import prefer_in_final_order

    results = [_result("a", 1.0)]
    assert prefer_in_final_order(results, frozenset({"zzz"})) is results
    assert prefer_in_final_order(results, frozenset()) is results


def test_preferred_in_reads_the_saved_project(db) -> None:
    from superlocalmemory.retrieval.project_scope import preferred_in
    from superlocalmemory.storage.models import RetrievalResult

    a = _save(db, "a", project="/x/acme")
    b = _save(db, "b")
    results = [RetrievalResult(fact=AtomicFact(fact_id=f, content=f)) for f in (a, b)]
    assert preferred_in(db, results, Facets.of(prefer_project="ACME")) == frozenset({a})
    assert preferred_in(db, results, Facets.of(project="acme")) == frozenset()
