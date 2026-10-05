# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Memories added around the search results never push a search result out.

After fusion, recall adds neighbours of the best results: every other memory
in the same scene, and bridge memories between two results. They are added
with a score taken from the result they came from, so one popular scene could
fill the whole pool the cross-encoder reads with its siblings, and a memory a
channel did find, a little further down, never reached the cross-encoder at
all. On the owner's store that was the cause of eval Q01 ("which release added
memory kinds"): the right memory was the 14th search result and was not in the
30 the cross-encoder scored, because 28 of those 30 were added neighbours.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.retrieval.fusion import FusionResult
from superlocalmemory.retrieval.rerank_pool import rerank_pool
from superlocalmemory.storage.models import AtomicFact

_SEEDS = [f"seed-{i:02d}" for i in range(12)]
_SIBLINGS = [f"sib-{i:02d}" for i in range(40)]
_GOLD = "gold"


def _fact(fid: str) -> AtomicFact:
    return AtomicFact(fact_id=fid, memory_id=f"m-{fid}", content=f"text of {fid}",
                      confidence=0.9)


def _db() -> MagicMock:
    facts = [_fact(f) for f in (*_SEEDS, *_SIBLINGS, _GOLD)]
    db = MagicMock()
    db.get_facts_by_ids.side_effect = (
        lambda ids, pid, **kw: [f for f in facts if f.fact_id in ids]
    )
    # One large scene around the best result, as a long session memory has.
    db.get_scenes_for_facts_batch.side_effect = lambda ids, pid: {
        _SEEDS[0]: [SimpleNamespace(fact_ids=list(_SIBLINGS))],
    }
    db.get_invalidated_fact_ids.return_value = set()
    db.get_nonapplied_correction_successor_ids.return_value = set()
    db.get_strict_temporal_excluded_fact_ids.return_value = set()
    return db


def _channel(results):
    ch = MagicMock()
    ch.search.return_value = results
    return ch


def _engine(reranker) -> RetrievalEngine:
    # The right memory is a real semantic hit, just below twelve others.
    semantic = [(fid, 0.9 - i * 0.01) for i, fid in enumerate(_SEEDS)] + [(_GOLD, 0.7)]
    entity = _channel([])
    # Siblings share the query's entities, so they also pass the evidence
    # floor - exactly what lets them through on a real store.
    entity.score_candidates.side_effect = (
        lambda q, ids, pid, **kw: {i: 0.5 for i in ids if i.startswith("sib-")}
    )
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    return RetrievalEngine(
        db=_db(), config=RetrievalConfig(),
        channels={"semantic": _channel(semantic), "entity_graph": entity},
        embedder=embedder, reranker=reranker,
    )


def _reranker_that_knows_the_answer() -> MagicMock:
    rr = MagicMock(spec=["rerank"])
    rr.rerank.side_effect = lambda q, cands, top_k: [
        (fact, 10.0 if fact.fact_id == _GOLD else 0.0) for fact, _ in cands
    ]
    return rr


class TestTheCrossEncoderSeesEverySearchResult:
    def test_a_found_memory_reaches_the_cross_encoder(self) -> None:
        rr = _reranker_that_knows_the_answer()
        _engine(rr).recall("which release added memory kinds", "default", limit=10)
        scored = {fact.fact_id for fact, _ in rr.rerank.call_args.args[1]}
        assert _GOLD in scored

    def test_so_the_cross_encoder_can_put_it_first(self) -> None:
        response = _engine(_reranker_that_knows_the_answer()).recall(
            "which release added memory kinds", "default", limit=10,
        )
        assert response.results[0].fact.fact_id == _GOLD

    def test_added_neighbours_still_reach_the_cross_encoder(self) -> None:
        rr = _reranker_that_knows_the_answer()
        _engine(rr).recall("which release added memory kinds", "default", limit=10)
        scored = {fact.fact_id for fact, _ in rr.rerank.call_args.args[1]}
        assert len(scored & set(_SIBLINGS)) >= 26  # what they held before

    def test_without_a_reranker_the_order_is_unchanged(self) -> None:
        response = _engine(None).recall(
            "which release added memory kinds", "default", limit=10,
        )
        ids = [r.fact.fact_id for r in response.results]
        assert ids[0] == _SEEDS[0]
        assert _GOLD not in ids  # fused order decides, as it always has


def _fr(fid: str, score: float) -> FusionResult:
    return FusionResult(fact_id=fid, fused_score=score)


class TestRerankPool:
    def test_keeps_the_best_of_each_kind_in_fused_order(self) -> None:
        fused = [_fr("e1", 0.9), _fr("c1", 0.8), _fr("e2", 0.7), _fr("e3", 0.6),
                 _fr("c2", 0.5), _fr("c3", 0.4)]
        pool = rerank_pool(fused, found={"c1", "c2", "c3"}, size=2)
        assert [fr.fact_id for fr in pool] == ["e1", "c1", "e2", "c2"]

    def test_is_a_superset_of_the_plain_cut(self) -> None:
        fused = [_fr(f"x{i}", 1.0 - i / 100) for i in range(50)]
        found = {f"x{i}" for i in range(0, 50, 3)}
        pool = rerank_pool(fused, found=found, size=10)
        assert [fr.fact_id for fr in pool[:10]] == [fr.fact_id for fr in fused[:10]]

    def test_nothing_added_means_the_plain_cut(self) -> None:
        fused = [_fr(f"c{i}", 1.0 - i / 100) for i in range(40)]
        pool = rerank_pool(fused, found={fr.fact_id for fr in fused}, size=30)
        assert pool == fused[:30]
