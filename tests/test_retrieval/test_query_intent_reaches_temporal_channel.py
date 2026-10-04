# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The time-aware channel must be told what kind of question it is answering.

The channel can only fall back to recency when it knows the question is about
the present. That knowledge is produced by the query classifier and has to
survive the trip through the engine's parallel channel dispatch to arrive at
the channel itself.

Both halves of that path are covered elsewhere: the classifier's output is
tested against a table of queries, and the channel's fallback is tested by
calling it directly with an explicit argument. Neither test would notice if
the engine dropped the argument in between, leaving the fallback permanently
unreachable in a running system while every unit test still passed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.storage.models import AtomicFact, Mode

# Every RetrievalEngine a test builds is closed when the test ends.
pytestmark = pytest.mark.usefixtures("closes_retrieval_engines")


class _TemporalSpy:
    """Stands in for the time-aware channel and records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query, profile_id, top_k=10, **kwargs):
        self.calls.append({"query": query, "kwargs": kwargs})
        return [("f_recent", 0.9)]


def _engine_with_temporal_spy(spy: _TemporalSpy) -> RetrievalEngine:
    fact = AtomicFact(
        fact_id="f_recent",
        memory_id="m0",
        content="the retrieval rewrite is the current focus",
        confidence=0.9,
    )
    db = MagicMock()
    db.get_facts_by_ids.side_effect = lambda ids, pid, **kw: [
        f for f in [fact] if f.fact_id in ids
    ]
    db.get_scenes_for_facts_batch.return_value = {}
    db.get_invalidated_fact_ids.return_value = set()
    db.get_nonapplied_correction_successor_ids.return_value = set()
    db.get_strict_temporal_excluded_fact_ids.return_value = set()

    bm = MagicMock()
    bm.search.return_value = [("f_recent", 1.2)]
    emb = MagicMock()
    emb.embed.return_value = [0.1, 0.2, 0.3]

    return RetrievalEngine(
        db=db,
        config=RetrievalConfig(),
        channels={"bm25": bm, "temporal": spy},
        embedder=emb,
    )


def _query_type_seen_by_channel(query: str) -> str | None:
    spy = _TemporalSpy()
    engine = _engine_with_temporal_spy(spy)
    engine.recall(query, "default", Mode.A, limit=10)
    assert spy.calls, f"the time-aware channel was never called for {query!r}"
    return spy.calls[0]["kwargs"].get("query_type")


def test_present_tense_question_arrives_as_recency() -> None:
    """A question about the present must reach the channel labelled as such.

    Without this the channel receives the default label, takes its early
    return, and contributes nothing to a question whose whole subject is time.
    """
    seen = _query_type_seen_by_channel("what am I working on")
    assert seen == "recency", (
        f"the channel was told {seen!r}; a present-tense question must arrive as "
        "'recency' or its recency fallback can never run in a live system"
    )


def test_factual_question_does_not_arrive_as_recency() -> None:
    """A plain lookup must not be labelled as a question about the present.

    If every question arrived as 'recency' the fallback would fire constantly
    and answer 'what is the architecture' with whatever was written today.

    The label must also actually be present. Asserting only that it is not
    'recency' would be satisfied by the engine passing no label at all, so
    this test would pass on a completely unwired pipeline.
    """
    seen = _query_type_seen_by_channel("what is the database schema")
    assert seen is not None, (
        "no label reached the channel at all; a test that only checks the label is "
        "not 'recency' would pass on an unwired pipeline, so it must check presence too"
    )
    assert seen != "recency", (
        "a factual lookup was labelled 'recency'; the recency fallback would fire "
        "on questions that have nothing to do with time"
    )


def test_the_label_is_passed_at_all() -> None:
    """The keyword must be present, whatever its value.

    Pinned separately so that removing the argument fails loudly here rather
    than silently reverting the channel to its default for every query.
    """
    spy = _TemporalSpy()
    engine = _engine_with_temporal_spy(spy)
    engine.recall("what happened in March", "default", Mode.A, limit=10)
    assert spy.calls, "the time-aware channel was never called"
    assert "query_type" in spy.calls[0]["kwargs"], (
        "the engine dispatched the time-aware channel without a query_type; the "
        "channel would silently use its default for every query in production"
    )
