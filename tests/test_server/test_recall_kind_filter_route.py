# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""GET /recall's ``kind`` filter (LLD/WP8 4.1.19).

``engine.recall`` itself is monkeypatched to a canned response (the reranker
and embedder are not under test here) so this exercises only the HTTP
boundary: validation before retrieval, the over-fetched ``limit`` the route
asks the engine for, and the filtered/truncated response shape.
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.storage.models import AtomicFact, FactType


def _result(fact_id, content, *, kind=None, source=None):
    fact = AtomicFact(fact_id=fact_id, memory_id=f"m-{fact_id}", content=content,
                      fact_type=FactType.SEMANTIC)
    if kind is not None:
        fact.memory_kind, fact.memory_kind_source = kind, source
    return SimpleNamespace(fact=fact, score=0.9, relevance_score=0.9, confidence=0.9,
                           memory_confidence=0.9, ranking_score=0.9, rank_position=1,
                           trust_score=0.5, channel_scores={}, evidence_chain=[])


def _response(results):
    return SimpleNamespace(
        results=results, query="q", query_type="lookup", retrieval_time_ms=5.0,
        channel_weights={}, total_candidates=len(results), no_confident_match=False,
    )


def test_recall_kind_filter_keeps_only_matches(engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    results = [
        _result("f1", "a decision", kind="decision", source="user"),
        _result("f2", "a fact", kind="semantic", source="user"),
        _result("f3", "another decision", kind="decision", source="user"),
    ]
    monkeypatch.setattr(engine_with_mock_deps, "recall", lambda *a, **k: _response(results))
    with _client(engine_with_mock_deps) as client:
        r = client.get("/recall", params={"q": "decisions", "kind": "decision"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {item["fact_id"] for item in body["results"]} == {"f1", "f3"}
    assert all(item["memory_kind"] == "decision" for item in body["results"])


def test_recall_kind_filter_overfetches_the_engine_call(engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    seen = {}

    def fake_recall(*args, **kwargs):
        seen.update(kwargs)
        seen["limit_arg"] = kwargs.get("limit")
        return _response([])

    monkeypatch.setattr(engine_with_mock_deps, "recall", fake_recall)
    with _client(engine_with_mock_deps) as client:
        client.get("/recall", params={"q": "decisions", "kind": "decision", "limit": 10})
        plain_seen = dict(seen)
        seen.clear()
        client.get("/recall", params={"q": "decisions", "limit": 10})
    assert plain_seen["limit_arg"] == 30  # overfetch_limit(10)
    assert seen["limit_arg"] == 10  # unchanged when no kind filter


def test_recall_rejects_an_unknown_kind_before_any_retrieval(engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    called = []
    monkeypatch.setattr(engine_with_mock_deps, "recall",
                        lambda *a, **k: called.append(1) or _response([]))
    with _client(engine_with_mock_deps) as client:
        r = client.get("/recall", params={"q": "decisions", "kind": "not-a-real-kind"})
    assert r.status_code == 422, r.text
    assert not called, "the engine must not be asked to retrieve for a kind that never parsed"
