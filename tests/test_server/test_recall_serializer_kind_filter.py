# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``serialize_recall_response``'s ``kind`` filter (LLD/WP8 4.1.19).

The filter runs on the DISPLAYED kind (what ``kind_fields`` computes, so a
legacy row filters by its mapped kind) over every candidate the caller
handed in — not just the first ``limit`` — and only THEN truncates to
``limit``. That ordering is what lets a caller over-fetch (see
``retrieval.kind_filter.overfetch_limit``) and still get up to ``limit``
matching results back.
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.server.recall_serializer import serialize_recall_response
from superlocalmemory.storage.models import AtomicFact, FactType


def _result(fact_id, content, *, kind=None, source=None, fact_type=FactType.SEMANTIC):
    fact = AtomicFact(fact_id=fact_id, content=content, fact_type=fact_type)
    if kind is not None:
        fact.memory_kind, fact.memory_kind_source = kind, source
    return SimpleNamespace(fact=fact, score=0.9, relevance_score=0.9, confidence=0.9,
                           memory_confidence=0.9, ranking_score=0.9, rank_position=1,
                           trust_score=0.5, channel_scores={}, evidence_chain=[])


def _response(results):
    return SimpleNamespace(results=results, query="q")


def test_no_kind_is_unfiltered_and_unchanged() -> None:
    results = [_result("f1", "a"), _result("f2", "b", kind="rule", source="user")]
    out, _ = serialize_recall_response(_response(results), limit=5)
    assert [r["fact_id"] for r in out] == ["f1", "f2"]


def test_kind_filter_keeps_only_matching_displayed_kind() -> None:
    results = [
        _result("f1", "a decision", kind="decision", source="user"),
        _result("f2", "a fact", kind="semantic", source="user"),
        _result("f3", "another decision", kind="decision", source="user"),
    ]
    out, _ = serialize_recall_response(_response(results), limit=5, kind="decision")
    assert [r["fact_id"] for r in out] == ["f1", "f3"]
    assert all(r["memory_kind"] == "decision" for r in out)


def test_kind_filter_matches_a_legacy_row_by_its_mapped_kind() -> None:
    # No memory_kind of its own; kind_fields() maps FactType.EPISODIC -> "episodic".
    results = [_result("f1", "something happened", fact_type=FactType.EPISODIC),
              _result("f2", "a fact", fact_type=FactType.SEMANTIC)]
    out, _ = serialize_recall_response(_response(results), limit=5, kind="episodic")
    assert [r["fact_id"] for r in out] == ["f1"]


def test_kind_filter_scans_beyond_the_first_limit_candidates() -> None:
    # 10 candidates fetched (the overfetch), the 2 matches sit past position
    # ``limit`` - the pre-filter behaviour of slicing to ``limit`` BEFORE
    # looking at kind would have missed both.
    results = [_result(f"f{i}", f"status {i}", kind="status", source="user")
              for i in range(8)]
    results += [_result("f_rule_a", "rule a", kind="rule", source="user"),
                _result("f_rule_b", "rule b", kind="rule", source="user")]
    out, _ = serialize_recall_response(_response(results), limit=2, kind="rule")
    assert {r["fact_id"] for r in out} == {"f_rule_a", "f_rule_b"}


def test_kind_filter_truncates_to_limit_after_filtering() -> None:
    results = [_result(f"f{i}", f"rule {i}", kind="rule", source="user") for i in range(5)]
    out, _ = serialize_recall_response(_response(results), limit=2, kind="rule")
    assert len(out) == 2


def test_empty_kind_string_behaves_like_no_filter() -> None:
    results = [_result("f1", "a", kind="rule", source="user")]
    out, _ = serialize_recall_response(_response(results), limit=5, kind="")
    assert [r["fact_id"] for r in out] == ["f1"]


def test_no_kind_filter_still_truncates_to_limit_when_more_candidates_exist() -> None:
    # Mirrors an aggregation query_type, where response.results can hold up
    # to 100 candidates regardless of the caller's limit. Without a kind
    # filter this must stay bounded to `limit` candidates examined, not
    # silently scan (and content-clamp/kind-compute) every one of them.
    results = [_result(f"f{i}", f"item {i}") for i in range(50)]
    out, _ = serialize_recall_response(_response(results), limit=3)
    assert [r["fact_id"] for r in out] == ["f0", "f1", "f2"]


def test_no_kind_filter_builds_at_most_limit_entries(monkeypatch) -> None:
    """The output-shape assertion above cannot tell "scan 50, keep 3" apart
    from "scan 3" — both return the same 3 ids. This counts the actual work:
    without a kind filter, kind_fields (called once per candidate inside the
    entry-building loop) must run exactly `limit` times, not once per
    over-sized response.results.
    """
    import superlocalmemory.server.recall_serializer as serializer_mod

    calls = []
    real_kind_fields = serializer_mod.kind_fields
    monkeypatch.setattr(serializer_mod, "kind_fields",
                        lambda *a, **k: calls.append(1) or real_kind_fields(*a, **k))

    results = [_result(f"f{i}", f"item {i}") for i in range(50)]
    serialize_recall_response(_response(results), limit=3)
    assert len(calls) == 3, f"expected exactly 3 entries built, kind_fields ran {len(calls)} times"
