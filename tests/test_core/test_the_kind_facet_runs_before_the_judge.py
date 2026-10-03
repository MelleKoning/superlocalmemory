# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""recall's ``kind`` filter is a facet, applied inside retrieval before the
answer check — not a post-serialization filter (4.1.19 WP8, defect found in
6722466b).

Why this matters: the answer check judges ``response.results[:top_k]``
(``DEFAULT_TOP_K = 3``). If ``kind`` were applied AFTER ``retrieval_engine.recall``
returns (as a post-serialization step, the original — wrong — design), the judge
would assess the UNFILTERED top three while the caller is shown a different,
filtered set: a receipt describing memories the caller never saw. Making
``kind`` a ``Facets`` field — matched via ``matching_fact_ids`` on the fused
candidate set, exactly where ``project``/``saved_by``/``about`` already run —
means ``response.results`` is already the filtered set by the time the judge
(or anything else downstream) ever sees it.

This test exercises ``core.recall_pipeline.run_recall`` with a stand-in
``retrieval_engine.recall`` that performs the SAME real filtering the
production ``RetrievalEngine.recall`` does (``matching_fact_ids`` against a
real, on-disk store) — so what is proven here is the composition: facets
reach the engine, matching_fact_ids honours them for real, and the judge is
handed only what results from that.
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.core import recall_pipeline
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval.facets import Facets, matching_fact_ids
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    FactType,
    MemoryRecord,
    RecallResponse,
    RetrievalResult,
)

_VERDICT = SufficiencyVerdict((0.9,), 0.5, "test:judge")


class _RecordingJudge:
    """Mirrors test_the_online_check_sends_only_this_profiles_memories._Judge."""

    top_k = 3
    rerank_enabled = False

    def __init__(self) -> None:
        self.backend = "jev"
        self.documents: list[str] = []

    def assess(self, query, documents, *, deadline=None):
        self.documents = [d.content for d in documents]
        return acs.JudgeOutcome(_VERDICT, acs.STATUS_JUDGED)

    def judge(self, query, documents):
        return self.assess(query, documents).verdict


def _db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _save(db, content, *, kind=None, source=None, fact_type=FactType.SEMANTIC) -> AtomicFact:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                      fact_type=fact_type, memory_kind=kind, memory_kind_source=source)
    fact_id = db.store_fact(fact)
    return db.get_facts_by_ids([fact_id], "default")[0]


def _real_facet_filtering_recall(db, facts, *, score_step=0.01):
    """A retrieval_engine.recall() stand-in that filters candidates through
    the REAL matching_fact_ids(), exactly where production code does — the
    thing actually under test is that run_recall forwards facets here, and
    that matching_fact_ids genuinely narrows by kind against the real store.
    """
    def _recall(query, profile_id, mode=None, limit=10, *, facets=None, **kwargs):
        fact_ids = [f.fact_id for f in facts]
        if facets is not None and not facets.empty:
            keep = matching_fact_ids(db, fact_ids, profile_id, facets)
            kept_facts = [f for f in facts if f.fact_id in keep]
        else:
            kept_facts = facts
        results = [
            RetrievalResult(fact=f, score=0.9 - i * score_step, confidence=1.0)
            for i, f in enumerate(kept_facts)
        ]
        return RecallResponse(query=query, results=results, query_type="lookup")
    return _recall


def _run_recall(db, facts, judge, config, monkeypatch, *, facets=None):
    from superlocalmemory.core import recall_pipeline as rp

    engine = SimpleNamespace(
        _sufficiency_judge=judge,
        recall=_real_facet_filtering_recall(db, facts),
    )
    monkeypatch.setattr(rp, "apply_ranking", lambda resp, *a, **k: resp)
    return rp.run_recall(
        "which deployment?", "default", fast=True, config=config, retrieval_engine=engine,
        trust_scorer=None, embedder=None, db=db, llm=None, hooks=None,
        facets=facets,
    )


def test_judge_sees_only_kind_matching_candidates(tmp_path, mode_a_config, monkeypatch) -> None:
    db = _db(tmp_path)
    decision_a = _save(db, "DECISION blue-green rollout.", kind="decision", source="user")
    status_a = _save(db, "STATUS currently on v4.1.18.", kind="status", source="user")
    decision_b = _save(db, "DECISION use SQLite for storage.", kind="decision", source="user")
    facts = [decision_a, status_a, decision_b]

    judge = _RecordingJudge()
    out = _run_recall(db, facts, judge, mode_a_config, monkeypatch,
                      facets=Facets.of(kind="decision"))

    assert judge.documents, "the judge was never asked"
    assert all("DECISION" in doc for doc in judge.documents), (
        f"the judge saw a non-matching candidate: {judge.documents}"
    )
    assert len(judge.documents) == 2
    # What the judge saw is exactly what the caller is shown — the whole point.
    assert {r.fact.content for r in out.results} == {d.content for d in [decision_a, decision_b]}


def test_without_a_kind_filter_the_judge_sees_everything_as_before(
    tmp_path, mode_a_config, monkeypatch,
) -> None:
    db = _db(tmp_path)
    facts = [_save(db, "a", kind="decision", source="user"), _save(db, "b", kind="status", source="user")]
    judge = _RecordingJudge()
    _run_recall(db, facts, judge, mode_a_config, monkeypatch, facets=None)
    assert sorted(judge.documents) == ["a", "b"]


# -- (2) kind AND project compose in one Facets ---------------------------------


def test_kind_and_project_compose_as_and(tmp_path) -> None:
    db = _db(tmp_path)

    def _save_with_project(content, *, project, kind):
        memory_id = db.store_memory(
            MemoryRecord(profile_id="default", content=content, metadata={"project": project}))
        fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                          fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source="user")
        return db.store_fact(fact)

    match = _save_with_project("A", project="slm", kind="decision")
    wrong_project = _save_with_project("B", project="other", kind="decision")
    wrong_kind = _save_with_project("C", project="slm", kind="status")
    ids = [match, wrong_project, wrong_kind]

    kept = matching_fact_ids(db, ids, "default", Facets.of(project="slm", kind="decision"))
    assert kept == {match}


# -- (3) a legacy row matches its mapped kind through the facet -----------------


def _facet_filtering_agentic_engine(db, facts):
    """A retrieval_engine stand-in with BOTH ``recall`` (round-1 entry) and
    ``recall_facts`` (what AgenticRetriever calls on every round) applying
    the SAME real ``matching_fact_ids`` filter — so this proves facets
    survive composition through the agentic (``fast=False``) path, not just
    the plain recall path ``_real_facet_filtering_recall`` above covers.
    """
    def _kept(facets):
        fact_ids = [f.fact_id for f in facts]
        if facets is not None and not facets.empty:
            keep = matching_fact_ids(db, fact_ids, "default", facets)
            return [f for f in facts if f.fact_id in keep]
        return list(facts)

    class _Engine:
        def recall(self, query, profile_id, mode=None, limit=10, *, facets=None, **kwargs):
            kept = _kept(facets)
            results = [
                RetrievalResult(fact=f, score=0.9 - i * 0.01, confidence=1.0)
                for i, f in enumerate(kept)
            ]
            return RecallResponse(query=query, results=results, query_type="multi_hop")

        def recall_facts(self, query, profile_id, top_k=20, skip_agentic=True, facets=None):
            kept = _kept(facets)
            return [(f, 0.9 - i * 0.01) for i, f in enumerate(kept)]

    return _Engine()


def test_facets_survive_the_internal_agentic_round_fast_false(
    tmp_path, mode_a_config, monkeypatch,
) -> None:
    """4.1.19 L2-08: with ``fast=False`` the internal agentic sufficiency
    round used to re-retrieve with NO facets at all and then REPLACE the
    already-filtered round-1 response with its unfiltered answer — a caller
    who asked for ``project=zephyr`` got results from other projects back,
    and the judge assessed that unfiltered set too. Multi-hop query_type
    always forces the agentic round regardless of score.
    """
    db = _db(tmp_path)

    def _save_with_project(content, *, project):
        memory_id = db.store_memory(
            MemoryRecord(profile_id="default", content=content, metadata={"project": project}))
        fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                          fact_type=FactType.SEMANTIC)
        fact_id = db.store_fact(fact)
        return db.get_facts_by_ids([fact_id], "default")[0]

    zephyr_a = _save_with_project("Zephyr deploy blocked on TLS", project="zephyr")
    zephyr_b = _save_with_project("Zephyr tabs over spaces", project="zephyr")
    atlas = _save_with_project("Atlas uses Postgres", project="atlas")
    facts = [zephyr_a, zephyr_b, atlas]

    engine = _facet_filtering_agentic_engine(db, facts)
    monkeypatch.setattr(recall_pipeline, "apply_ranking", lambda resp, *a, **k: resp)
    out = recall_pipeline.run_recall(
        "Zephyr deploy and Atlas Postgres", "default", fast=False,
        config=mode_a_config, retrieval_engine=engine,
        trust_scorer=None, embedder=None, db=db, llm=None, hooks=None,
        facets=Facets.of(project="zephyr"),
    )

    projects = {r.fact.content for r in out.results}
    assert projects == {zephyr_a.content, zephyr_b.content}, projects
    assert atlas.content not in projects


def test_a_legacy_row_matches_its_mapped_kind(tmp_path) -> None:
    db = _db(tmp_path)
    # No memory_kind of its own; kind_fields() maps FactType.EPISODIC -> "episodic".
    legacy = _save(db, "something happened", fact_type=FactType.EPISODIC)
    other = _save(db, "a fact", fact_type=FactType.SEMANTIC)
    ids = [legacy.fact_id, other.fact_id]

    assert matching_fact_ids(db, ids, "default", Facets.of(kind="episodic")) == {legacy.fact_id}
    assert matching_fact_ids(db, ids, "default", Facets.of(kind="semantic")) == {other.fact_id}
