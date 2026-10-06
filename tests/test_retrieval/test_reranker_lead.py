# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Layers after the cross-encoder break its near-ties, never its clear leads."""

from __future__ import annotations

from superlocalmemory.core import recall_pipeline
from superlocalmemory.retrieval.reranker_lead import LEAD_BOUND, hold_reranker_lead
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult


def _r(fid: str, ranking: float) -> RetrievalResult:
    return RetrievalResult(fact=AtomicFact(fact_id=fid, content=fid), ranking_score=ranking)


def _ids(results) -> list[str]:
    return [r.fact.fact_id for r in results]


def test_a_clear_lead_is_restored() -> None:
    # Later layers put b first; the reranker scored a 2x higher.
    out = hold_reranker_lead([_r("b", 0.9), _r("a", 0.5)], {"a": 0.8, "b": 0.4})
    assert _ids(out) == ["a", "b"]


def test_a_near_tie_is_left_to_the_later_layers() -> None:
    near = 0.8 / (LEAD_BOUND * 0.99)  # b trails a by just under the bound
    out = hold_reranker_lead([_r("b", 0.9), _r("a", 0.5)], {"a": 0.8, "b": near})
    assert _ids(out) == ["b", "a"]


def test_the_bound_is_exactly_the_factor() -> None:
    at = {"a": 1.0, "b": 1.0 / LEAD_BOUND}
    assert _ids(hold_reranker_lead([_r("b", 2), _r("a", 1)], at)) == ["b", "a"]
    past = {"a": 1.0, "b": 1.0 / LEAD_BOUND - 1e-6}
    assert _ids(hold_reranker_lead([_r("b", 2), _r("a", 1)], past)) == ["a", "b"]


def test_only_what_is_blocked_moves() -> None:
    # c leads d clearly; a and b are a near-tie the later layers flipped.
    results = [_r("b", 4), _r("a", 3), _r("d", 2), _r("c", 1)]
    anchors = {"a": 0.80, "b": 0.75, "c": 0.50, "d": 0.10}
    assert _ids(hold_reranker_lead(results, anchors)) == ["b", "a", "c", "d"]


def test_no_result_ever_stands_above_a_clear_leader() -> None:
    import random

    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(2, 12)
        anchors = {f"f{i}": rng.random() for i in range(n)}
        results = [_r(f"f{i}", rng.random()) for i in rng.sample(range(n), n)]
        out = hold_reranker_lead(results, anchors)
        assert sorted(_ids(out)) == sorted(anchors)
        for hi, upper in enumerate(out):
            for lower in out[hi + 1:]:
                assert not (anchors[lower.fact.fact_id]
                            > LEAD_BOUND * anchors[upper.fact.fact_id])


def test_ranking_scores_fall_with_the_new_order() -> None:
    out = hold_reranker_lead([_r("b", 0.9), _r("a", 0.5)], {"a": 0.8, "b": 0.4})
    assert [r.ranking_score for r in out] == [0.9, 0.5]


def test_without_reranker_scores_nothing_moves() -> None:
    results = [_r("b", 0.9), _r("a", 0.5)]
    assert hold_reranker_lead(results, {}) == results
    assert hold_reranker_lead(results, {"a": 0.8}) == results  # b has none


def test_the_input_is_not_modified() -> None:
    results = [_r("b", 0.9), _r("a", 0.5)]
    hold_reranker_lead(results, {"a": 0.8, "b": 0.4})
    assert _ids(results) == ["b", "a"] and results[0].ranking_score == 0.9


def _recall_through_the_pipeline(monkeypatch, tmp_path, results, learned):
    from dataclasses import replace
    from types import SimpleNamespace

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.retrieval = replace(config.retrieval, sufficiency_judge="off")
    engine = SimpleNamespace(recall=lambda *a, **k: RecallResponse(
        query="q", results=list(results), query_type="factual"))
    monkeypatch.setattr(recall_pipeline, "apply_ranking", learned)
    return recall_pipeline.run_recall(
        "q", "default", fast=True, config=config, retrieval_engine=engine,
        trust_scorer=None, embedder=None, db=SimpleNamespace(db_path=None),
        llm=None, hooks=None)


def _reverse(response, *args, **kwargs):
    # A learned layer that rebuilds the response in the opposite order.
    return RecallResponse(query=response.query, results=list(reversed(response.results)),
                          query_type=response.query_type)


def _scored(fid: str, rerank: float | None) -> RetrievalResult:
    return RetrievalResult(fact=AtomicFact(fact_id=fid, content=f"memory {fid}"),
                           ranking_score=rerank or 0.5, rerank_score=rerank)


def test_recall_holds_a_clear_lead_after_the_learned_layers(monkeypatch, tmp_path) -> None:
    out = _recall_through_the_pipeline(
        monkeypatch, tmp_path, [_scored("a", 0.8), _scored("b", 0.3)], _reverse)
    assert _ids(out.results) == ["a", "b"]


def test_recall_lets_the_learned_layers_break_a_near_tie(monkeypatch, tmp_path) -> None:
    out = _recall_through_the_pipeline(
        monkeypatch, tmp_path, [_scored("a", 0.8), _scored("b", 0.79)], _reverse)
    assert _ids(out.results) == ["b", "a"]


def test_recall_without_the_reranker_is_unchanged(monkeypatch, tmp_path) -> None:
    out = _recall_through_the_pipeline(
        monkeypatch, tmp_path, [_scored("a", None), _scored("b", None)], _reverse)
    assert _ids(out.results) == ["b", "a"]
