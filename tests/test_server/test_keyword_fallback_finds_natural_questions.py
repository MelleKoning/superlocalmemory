# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The over-budget keyword answer must find what a person asks, and say it
skipped everything else (4.1.20 WP10).

It matched the whole question as one substring, so "When is the Halcyon
migration window?" could never match "The migration window for Project
Halcyon is 14 March" — and it reported ``channel_status == {}`` and
``incomplete_channels == []``, the shape of a complete recall that found
nothing.
"""

from __future__ import annotations

from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.server.recall_fallback import fallback_terms
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


def _save(engine, content):
    mid = engine._db.store_memory(
        MemoryRecord(profile_id=engine.profile_id, content=content),
    )
    return engine._db.store_fact(
        AtomicFact(profile_id=engine.profile_id, memory_id=mid, content=content,
                   fact_type=FactType.SEMANTIC),
    )


def test_a_natural_question_finds_the_saved_sentence(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    target = _save(engine_with_mock_deps,
                   "The migration window for Project Halcyon is 14 March")
    _save(engine_with_mock_deps, "Lunch order: two coffees")

    out = _recall_keyword_fallback(
        engine_with_mock_deps, "When is the Halcyon migration window?", 5)

    assert [r["fact_id"] for r in out["results"]] == [target]


def test_rows_with_more_of_the_question_rank_first(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    partial = _save(engine_with_mock_deps, "Halcyon has a new logo")
    full = _save(engine_with_mock_deps, "Halcyon migration window opens Friday")

    out = _recall_keyword_fallback(
        engine_with_mock_deps, "Halcyon migration window", 5)

    assert [r["fact_id"] for r in out["results"]] == [full, partial]


def test_the_envelope_says_every_channel_was_skipped(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    out = _recall_keyword_fallback(engine_with_mock_deps, "anything", 5)

    assert out["retrieval_mode"] == "degraded_lexical"
    assert set(out["channel_status"]) == set(chstat.CHANNEL_NAMES)
    assert all(v == chstat.TIMEOUT for v in out["channel_status"].values())
    assert out["incomplete_channels"] == sorted(chstat.CHANNEL_NAMES)


def test_like_wildcards_in_the_question_are_literal(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    _save(engine_with_mock_deps, "split the budget 50x50")
    literal = _save(engine_with_mock_deps, "split the budget 50_50")

    out = _recall_keyword_fallback(engine_with_mock_deps, "50_50", 5)

    # Unescaped, "_" matches any character and "50x50" would come back too.
    assert [r["fact_id"] for r in out["results"]] == [literal]


def test_terms_are_bounded_and_skip_filler_words() -> None:
    terms = fallback_terms("When is the Halcyon migration window?")
    assert terms[0] == "When is the Halcyon migration window?"
    assert terms[1:] == ["halcyon", "migration", "window"]
    many = fallback_terms(" ".join(f"word{i}" for i in range(40)))
    assert len(many) <= 8
    assert fallback_terms("   ") == []
