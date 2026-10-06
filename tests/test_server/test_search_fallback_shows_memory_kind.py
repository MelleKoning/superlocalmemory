# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``POST /api/search``'s degraded-lexical fallback (engine unavailable or
over its recall time budget) must show the memory kind the same way the
primary (engine) path does.

WHY THIS TEST EXISTS
---------------------
The primary path attaches kind_fields() via server.recall_serializer's
serialize_recall_response(). The fallback — a raw SQL keyword match used
when the engine is slow or down — built its result dicts straight from the
row, with at most the bare ``memory_kind`` column (no legacy-fact_type
fallback, no confirmed/suggested-threshold precedence). A user searching
in degraded mode would see a DIFFERENT (or missing) kind for the exact same
memory than they would once the engine recovered — "the same way recall
does" is specifically about not disagreeing with yourself depending on
load.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": d.capability,
        "X-SLM-Target-Instance": d.instance_id,
    }


def _slow_recall(*_a, **_k):
    time.sleep(1.0)
    raise RuntimeError("never reached in time — forces the degraded_lexical fallback")


@pytest.fixture()
def client(engine_with_mock_deps, monkeypatch):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    db = engine._db
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default','default')"
    )

    # fact-confirmed: a user-confirmed "decision" kind, legacy fact_type
    # "episodic" — if the fallback regressed to showing fact_type, this
    # would read "episodic" instead of "decision".
    mid1 = db.store_memory(MemoryRecord(profile_id="default", content="source"))
    db.store_fact(AtomicFact(
        fact_id="fact-confirmed", profile_id="default", memory_id=mid1,
        content="zorblatt decided we ship on Tuesdays",
        fact_type=FactType.EPISODIC, memory_kind="decision", memory_kind_source="user",
    ))

    # fact-legacy-only: no memory_kind at all — pure pre-4.1.19 row, must
    # fall back to the legacy-mapped kind, not come back untyped.
    mid2 = db.store_memory(MemoryRecord(profile_id="default", content="source"))
    db.store_fact(AtomicFact(
        fact_id="fact-legacy-only", profile_id="default", memory_id=mid2,
        content="zorblatt is a type of cryptocurrency",
        fact_type=FactType.OPINION,
    ))

    monkeypatch.setattr(engine, "recall", _slow_recall)
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.2")

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    yield TestClient(app), _daemon_headers(app)


def _by_id(payload: dict) -> dict[str, dict]:
    return {r["fact_id"]: r for r in payload.get("results", [])}


class TestDegradedSearchShowsTheMemoryKind:
    def test_it_actually_took_the_fallback_path(self, client) -> None:
        tc, h = client
        r = tc.post("/api/search", json={"query": "zorblatt", "limit": 10}, headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["retrieval_mode"] == "degraded_lexical", r.json()

    def test_a_confirmed_kind_wins_over_the_legacy_fact_type(self, client) -> None:
        tc, h = client
        r = tc.post("/api/search", json={"query": "zorblatt", "limit": 10}, headers=h)
        row = _by_id(r.json())["fact-confirmed"]
        assert row["memory_kind"] == "decision", row
        assert row["memory_kind_label"] == "Decision", row
        assert row["memory_kind_state"] == "confirmed", row

    def test_a_row_with_no_kind_data_shows_the_legacy_mapped_kind(self, client) -> None:
        tc, h = client
        r = tc.post("/api/search", json={"query": "zorblatt", "limit": 10}, headers=h)
        row = _by_id(r.json())["fact-legacy-only"]
        assert row["memory_kind"] == "opinion", row
        assert row["memory_kind_label"] == "Preference or view", row
        assert row["memory_kind_state"] == "legacy", row

    def test_the_kind_filter_still_narrows_the_fallback_too(self, client) -> None:
        tc, h = client
        r = tc.post(
            "/api/search",
            json={"query": "zorblatt", "limit": 10, "kind": "decision"},
            headers=h,
        )
        assert r.status_code == 200, r.text
        assert list(_by_id(r.json())) == ["fact-confirmed"]
