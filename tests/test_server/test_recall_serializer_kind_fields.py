# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Every recall result says what kind of memory it is, the same way on every surface."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.server.recall_serializer import serialize_recall_response
from superlocalmemory.storage.models import AtomicFact, FactType

KIND_KEYS = {"memory_kind", "memory_kind_label", "memory_kind_state",
             "memory_kind_source", "memory_kind_confidence"}


def _response(fact):
    result = SimpleNamespace(fact=fact, score=0.9, relevance_score=0.9, confidence=0.9,
                             memory_confidence=0.9, ranking_score=0.9, rank_position=1,
                             trust_score=0.5, channel_scores={}, evidence_chain=[])
    return SimpleNamespace(results=[result], query="q")


def _first(fact) -> dict:
    results, _truncated = serialize_recall_response(_response(fact), limit=5)
    return results[0]


def test_a_confirmed_rule_is_labelled_as_one() -> None:
    fact = AtomicFact(fact_id="f1", content="Always run the suite once.", fact_type=FactType.SEMANTIC)
    fact.memory_kind, fact.memory_kind_source = "rule", "caller"
    entry = _first(fact)
    assert KIND_KEYS <= set(entry)
    assert entry["memory_kind"] == "rule" and entry["memory_kind_state"] == "confirmed"
    assert entry["memory_kind_label"] == "Standing rule"


def test_an_untyped_memory_still_has_every_field() -> None:
    entry = _first(AtomicFact(fact_id="f2", content="Something.", fact_type=FactType.EPISODIC))
    assert KIND_KEYS <= set(entry)
    assert entry["memory_kind_state"] in ("untyped", "legacy")


def test_the_caller_s_configured_threshold_is_honoured_not_a_hard_coded_one() -> None:
    """4.1.21 #16: every surface that labels a kind reads the SAME configured
    ``memory_kinds.display_min_confidence`` (``slm list``, MCP, the kind
    facet) — this chokepoint must not keep its own 0.20 regardless of what a
    caller with a live ``SLMConfig`` passes in."""
    fact = AtomicFact(fact_id="f3", content="Ship on Fridays only with sign-off.",
                       fact_type=FactType.SEMANTIC)
    fact.memory_kind, fact.memory_kind_source, fact.memory_kind_confidence = (
        "rule", "model", 0.30)
    # 0.30 clears the module default (0.20): shown as a suggestion when the
    # caller passes nothing (today's behaviour, unchanged).
    entry = _first(fact)
    assert entry["memory_kind"] == "rule" and entry["memory_kind_state"] == "suggested"
    # A caller with a stricter configured threshold (0.50) must see the SAME
    # fact demoted to its legacy fact_type mapping — never "rule" at 0.30
    # confidence dressed up as a suggestion because this function ignored
    # what was configured.
    results, _truncated = serialize_recall_response(_response(fact), limit=5,
                                                     display_min_confidence=0.50)
    stricter = results[0]
    assert stricter["memory_kind_state"] == "legacy"
    assert stricter["memory_kind"] == "semantic"
    assert stricter["memory_kind_label"] == "Fact"
