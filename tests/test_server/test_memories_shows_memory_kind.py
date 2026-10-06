# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""``GET /api/memories`` must show the memory kind, not the legacy fact_type.

WHY THIS TEST EXISTS
---------------------
4.1.19 introduced nine memory kinds (storage/memory_kinds.py) with a shared,
confidence-threshold-aware serializer, ``kind_fields()`` — used by `slm list`,
recall, and the MCP tools. ``GET /api/memories`` (the dashboard's Memories ->
All memories table) never called it: it returned the raw legacy ``fact_type``
(aliased ``category``) and nothing else, so the CATEGORY column showed
"semantic" / "episodic" / "opinion" for every row regardless of what kind
(if any) a classifier had confidently assigned it, and the kind filter chips
had no server-side filter to call.

This pins three things kind_fields is responsible for:
  * a CONFIRMED kind (source in CONFIRMED_SOURCES) is shown over the legacy
    fact_type;
  * a row with no confident kind falls back to the legacy-mapped kind (not
    "untyped", not the raw fact_type string) — the same fallback `slm list`
    and recall use;
  * the new `kind=` filter narrows on the DISPLAYED kind, the same way
    core/kind_query.py's `list_recent_facts`/`search_facts` do for MCP/CLI.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": d.capability,
        "X-SLM-Target-Instance": d.instance_id,
    }


@pytest.fixture()
def client(engine_with_mock_deps):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default','default')"
    )
    engine._db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        " speaker, role, created_at, metadata_json, scope) "
        "VALUES ('m1','default','source','s1','user','user',"
        " '2026-01-01T00:00:00Z','{}','personal')"
    )
    # fact-confirmed: a user-confirmed "decision" kind. fact_type is the
    # legacy "episodic" — if the route ever regresses to showing fact_type,
    # this would read "episodic" instead of "decision".
    engine._db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        " lifecycle, created_at, scope, quarantined, fact_type, "
        " memory_kind, memory_kind_source, memory_kind_confidence) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("fact-confirmed", "m1", "default", "We ship Tuesdays.",
         "active", "2026-01-01T00:00:00Z", "personal", 0,
         "episodic", "decision", "user", None),
    )
    # fact-below-threshold: a model suggestion with confidence BELOW the
    # configured display threshold — must fall back to the legacy kind, not
    # show the low-confidence suggestion.
    engine._db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        " lifecycle, created_at, scope, quarantined, fact_type, "
        " memory_kind, memory_kind_source, memory_kind_confidence) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("fact-below-threshold", "m1", "default", "ledger is a type of cryptocurrency",
         "active", "2026-01-01T00:00:00Z", "personal", 0,
         "semantic", "rule", "model:llm", 0.05),
    )
    # fact-legacy-only: no memory_kind at all — pure pre-4.1.19 row.
    engine._db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        " lifecycle, created_at, scope, quarantined, fact_type) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("fact-legacy-only", "m1", "default", "The harness caught two vacuous tests.",
         "active", "2026-01-01T00:00:00Z", "personal", 0, "opinion"),
    )

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    yield TestClient(app), _daemon_headers(app)


def _by_id(payload: dict) -> dict[str, dict]:
    return {m["id"]: m for m in payload.get("memories", [])}


class TestTheCategoryColumnShowsTheMemoryKind:
    def test_a_confirmed_kind_wins_over_the_legacy_fact_type(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50", headers=h)
        assert r.status_code == 200, r.text
        row = _by_id(r.json())["fact-confirmed"]
        assert row["memory_kind"] == "decision", row
        assert row["memory_kind_label"] == "Decision", row
        assert row["memory_kind_state"] == "confirmed", row

    def test_a_suggestion_below_threshold_falls_back_to_the_legacy_kind(
        self, client,
    ) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50", headers=h)
        row = _by_id(r.json())["fact-below-threshold"]
        # fact_type "semantic" legacy-maps to MemoryKind.SEMANTIC ("Fact"),
        # NOT the low-confidence "rule" suggestion.
        assert row["memory_kind"] == "semantic", row
        assert row["memory_kind_state"] == "legacy", row

    def test_a_row_with_no_kind_data_shows_the_legacy_mapped_kind(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50", headers=h)
        row = _by_id(r.json())["fact-legacy-only"]
        assert row["memory_kind"] == "opinion", row
        assert row["memory_kind_label"] == "Preference or view", row
        assert row["memory_kind_state"] == "legacy", row


class TestTheKindCountsEndpoint:
    """Backs the filter chips — one count per DISPLAYED kind, not per legacy
    fact_type (four buckets could never show nine chips with real counts)."""

    def test_counts_are_keyed_by_displayed_kind(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories/kind-counts", headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["counts"] == {"decision": 1, "semantic": 1, "opinion": 1}, body
        assert body["truncated"] is False


class TestTheKindFilterNarrowsOnTheDisplayedKind:
    def test_filtering_by_kind_returns_only_that_kind(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50&kind=decision", headers=h)
        assert r.status_code == 200, r.text
        ids = list(_by_id(r.json()))
        assert ids == ["fact-confirmed"], ids

    def test_filtering_by_the_legacy_fallback_kind_still_matches(self, client) -> None:
        """fact-below-threshold and fact-legacy-only both display as "semantic"
        / "opinion" respectively via the legacy fallback — the filter must
        match what is SHOWN, not the raw (absent or low-confidence) column."""
        tc, h = client
        r = tc.get("/api/memories?limit=50&kind=semantic", headers=h)
        assert list(_by_id(r.json())) == ["fact-below-threshold"]

    def test_an_unknown_kind_is_rejected_not_silently_ignored(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50&kind=not-a-real-kind", headers=h)
        assert r.status_code == 400, r.text

    def test_total_and_has_more_agree_with_the_filtered_rows(self, client) -> None:
        tc, h = client
        body = tc.get("/api/memories?limit=50&kind=opinion", headers=h).json()
        assert body["total"] == 1
        assert len(body["memories"]) == 1
        assert body["has_more"] is False
