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
