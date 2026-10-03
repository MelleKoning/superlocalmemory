# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall uses a memory's kind for the question asked - within strict bounds."""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.kind_aware import apply_kind_awareness, query_intent
from superlocalmemory.storage.memory_kinds import MemoryKind
from superlocalmemory.storage.models import AtomicFact, FactType, RetrievalResult


def _r(fid, score, kind=None, source="caller", created="2026-10-01", entities=("e_slm",)):
    fact = AtomicFact(fact_id=fid, content=fid, fact_type=FactType.SEMANTIC,
                      created_at=created, canonical_entities=list(entities))
    fact.memory_kind, fact.memory_kind_source = kind, (source if kind else None)
    return RetrievalResult(fact=fact, score=score)


def _ids(results):
    return [r.fact.fact_id for r in results]


@pytest.mark.parametrize("query,kinds", [
    ("What did we decide about the recall ceiling?", {MemoryKind.DECISION}),
    ("How do I upgrade SLM with pipx?", {MemoryKind.PROCEDURE}),
    ("What is the current status of 4.1.19?", {MemoryKind.STATUS}),
    ("What are the rules for releases?", {MemoryKind.RULE}),
    ("What's next on the plan?", {MemoryKind.PROSPECTIVE}),
    ("Which editor do I prefer?", {MemoryKind.OPINION}),
    ("Who maintains the bounded loops server?", set()),
    ("Tell me about Laya", set()),
])
def test_intent_is_read_from_the_question(query, kinds) -> None:
    assert set(query_intent(query).kinds) == kinds


def test_a_question_without_intent_keeps_its_exact_order() -> None:
    results = [_r("a", 0.9), _r("b", 0.8, "decision"), _r("c", 0.7, "rule")]
    assert _ids(apply_kind_awareness(results, "Tell me about Laya")) == ["a", "b", "c"]


def test_a_matching_kind_passes_a_close_neighbour_only() -> None:
    results = [_r("close", 0.90), _r("far", 1.20), _r("decision", 0.85, "decision")]
    results.sort(key=lambda r: -r.score)                          # far, close, decision
    out = _ids(apply_kind_awareness(results, "what did we decide about SLM"))
    assert out == ["far", "decision", "close"]                    # passes 0.90, never 1.20


def test_a_suggested_kind_counts_half() -> None:
    confirmed = [_r("plain", 0.90), _r("d", 0.80, "decision", "caller")]
    suggested = [_r("plain", 0.90), _r("d", 0.80, "decision", "model:laya")]
    q = "what did we decide"
    assert _ids(apply_kind_awareness(confirmed, q)) == ["d", "plain"]    # 0.80*1.15 > 0.90
    assert _ids(apply_kind_awareness(suggested, q)) == ["plain", "d"]    # 0.80*1.075 < 0.90


def test_the_newest_current_state_comes_first_and_the_older_says_so() -> None:
    old = _r("status-old", 0.95, "status", created="2026-09-01")
    new = _r("status-new", 0.60, "status", created="2026-10-02")
    other = _r("other-subject", 0.50, "status", created="2026-10-03", entities=("e_other",))
    out = apply_kind_awareness([old, new, other], "what is the current status of SLM")
    assert _ids(out) == ["status-new", "status-old", "other-subject"]
    assert "newer:status-new" in out[1].evidence_chain
    assert [r.rank_position for r in out] == [1, 2, 3]


def test_latest_first_never_mixes_kinds_or_subjects() -> None:
    decision = _r("decision-old", 0.9, "decision", created="2026-09-01")
    status = _r("status-new", 0.5, "status", created="2026-10-01")
    out = apply_kind_awareness([decision, status], "current status and what did we decide")
    assert _ids(out) == ["decision-old", "status-new"]


def test_nothing_is_added_or_removed_and_disabled_means_untouched() -> None:
    results = [_r(f"f{i}", 1.0 - i * 0.01, "decision" if i % 2 else None) for i in range(14)]
    out = apply_kind_awareness(list(results), "what did we decide")
    assert sorted(_ids(out)) == sorted(_ids(results))
    assert _ids(out)[10:] == _ids(results)[10:]                   # below the head: untouched
    assert _ids(apply_kind_awareness(list(results), "what did we decide", enabled=False)) \
        == _ids(results)
