# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L3-20: POST /api/search takes the same ``kind`` filter MCP's ``search``
and ``list_recent`` already take, validated the same way (422,
{"code": "INVALID_KIND", ...}) and applied to whichever path answers: the
engine's semantic recall, or the degraded-lexical DB fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.models import AtomicFact, FactType


def _result(fact_id, content, *, kind):
    fact = AtomicFact(fact_id=fact_id, memory_id=f"m-{fact_id}", content=content,
                      fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source="user")
    return SimpleNamespace(fact=fact, score=0.9, relevance_score=0.9, confidence=0.9,
                           memory_confidence=0.9, ranking_score=0.9, rank_position=1,
                           trust_score=0.5, channel_scores={}, evidence_chain=[])


_CANDIDATES = [
    _result("f1", "a decision", kind="decision"),
    _result("f2", "a rule", kind="rule"),
    _result("f3", "another decision", kind="decision"),
]


def _canned_recall(*_args, **_kwargs):
    return SimpleNamespace(results=_CANDIDATES, query="q", query_type="lookup",
                           retrieval_time_ms=1.0, channel_weights={},
                           total_candidates=len(_CANDIDATES), no_confident_match=False)


def _client(engine_with_mock_deps) -> TestClient:
    engine = engine_with_mock_deps
    engine.recall = _canned_recall
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    return TestClient(app)


def test_search_rejects_an_unknown_kind(engine_with_mock_deps) -> None:
    r = _client(engine_with_mock_deps).post(
        "/api/search", json={"query": "anything", "limit": 10, "kind": "not-a-real-kind"})
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "INVALID_KIND"


def test_search_kind_filter_narrows_results(engine_with_mock_deps) -> None:
    client = _client(engine_with_mock_deps)
    r = client.post("/api/search", json={"query": "decisions", "limit": 10, "kind": "decision"})
    assert r.status_code == 200, r.text
    ids = {item["fact_id"] for item in r.json()["results"]}
    assert ids == {"f1", "f3"}


def test_search_without_a_kind_is_unfiltered(engine_with_mock_deps) -> None:
    client = _client(engine_with_mock_deps)
    r = client.post("/api/search", json={"query": "decisions", "limit": 10})
    assert r.status_code == 200, r.text
    ids = {item["fact_id"] for item in r.json()["results"]}
    assert ids == {"f1", "f2", "f3"}
