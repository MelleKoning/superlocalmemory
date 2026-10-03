# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""GET /recall's ``kind`` filter (LLD/WP8 4.1.19).

``kind`` is a facet (retrieval/facets.py), matched inside retrieval before the
answer check — not a post-serialization filter. A defect in 6722466b over-fetched
and filtered AFTER ``engine.recall`` returned, so the answer check judged the
unfiltered top three while the caller saw a different, filtered set. See
tests/test_core/test_the_kind_facet_runs_before_the_judge.py for the test that
proves the judge and the caller now see the same memories; this file only
covers the HTTP boundary, mirroring test_retrieval/test_recall_facets.py's
``test_http_recall_passes_facets_to_the_engine``.
"""

from __future__ import annotations

from types import SimpleNamespace


def _response(results):
    return SimpleNamespace(
        results=results, query="q", query_type="lookup", retrieval_time_ms=5.0,
        channel_weights={}, total_candidates=len(results), no_confident_match=False,
    )


def test_recall_passes_kind_as_a_facet_with_the_original_limit(
    engine_with_mock_deps, monkeypatch,
) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    seen = {}

    def fake_recall(*args, **kwargs):
        seen.update(kwargs)
        return _response([])

    monkeypatch.setattr(engine_with_mock_deps, "recall", fake_recall)
    with _client(engine_with_mock_deps) as client:
        client.get("/recall", params={"q": "decisions", "kind": "decision", "limit": 10})
        plain_seen = dict(seen)
        seen.clear()
        client.get("/recall", params={"q": "decisions", "limit": 10})

    assert plain_seen["facets"].kind == "decision"
    # No over-fetch: the facet is matched inside retrieval, so the engine is
    # asked for exactly what the caller asked for, same as every other facet.
    assert plain_seen["limit"] == 10
    assert "facets" not in seen
    assert seen["limit"] == 10


def test_recall_rejects_an_unknown_kind_before_any_retrieval(
    engine_with_mock_deps, monkeypatch,
) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    called = []
    monkeypatch.setattr(engine_with_mock_deps, "recall",
                        lambda *a, **k: called.append(1) or _response([]))
    with _client(engine_with_mock_deps) as client:
        r = client.get("/recall", params={"q": "decisions", "kind": "not-a-real-kind"})
    assert r.status_code == 422, r.text
    assert not called, "the engine must not be asked to retrieve for a kind that never parsed"
