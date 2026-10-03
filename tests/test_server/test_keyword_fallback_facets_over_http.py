# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""End-to-end: when ``GET /recall`` falls back to the over-budget keyword
search, the ``kind``/``project``/``saved_by``/``about`` query params it was
given still apply — exercised through the real HTTP route, not just the
``_recall_keyword_fallback`` unit.

4.1.19 L2-09 / L3-10 (lane 3's ``test_fallback_facets.py`` reproduced this at
the HTTP layer: its assertion recorded the BUG — that a project=apollo
"status" memory leaked into a project=zeus kind=decision request. This test
asserts the fixed, correct behaviour instead of the bug.
"""

from __future__ import annotations

import time

from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from tests.test_server.test_canonical_remember_route import _client


def test_keyword_fallback_honours_kind_and_project_over_http(
    engine_with_mock_deps, monkeypatch,
) -> None:
    engine = engine_with_mock_deps
    db = engine._db
    for content, kind, project in [
        ("deploy decision: we deploy on Thursday", "decision", "zeus"),
        ("deploy status: the deploy is broken right now", "status", "apollo"),
    ]:
        mid = db.store_memory(MemoryRecord(
            profile_id=engine.profile_id, content=content, metadata={"project": project},
        ))
        db.store_fact(AtomicFact(
            profile_id=engine.profile_id, memory_id=mid, content=content,
            fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source="user",
        ))

    def slow_recall(*a, **k):
        time.sleep(1.0)
        raise RuntimeError("never reached in time")

    monkeypatch.setattr(engine, "recall", slow_recall)
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.2")
    with _client(engine) as client:
        r = client.get("/recall", params={"q": "deploy", "kind": "decision", "project": "zeus"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["retrieval_mode"] == "degraded_lexical"
    assert body["facet_filter_error"] is None
    contents = [i["content"] for i in body["results"]]
    assert any("decision" in c for c in contents)
    assert not any("status" in c for c in contents), contents
