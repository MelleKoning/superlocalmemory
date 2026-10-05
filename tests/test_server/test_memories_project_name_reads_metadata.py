# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""``GET /api/memories`` must show the project a memory was saved under.

WHY THIS TEST EXISTS
---------------------
``remember(project=...)`` (and ``slm remember --project``) store the project
in ``memories.metadata_json -> '$.project'`` (see
``core/project_identity.py::storable_project`` and
``mcp/tools_core.py``'s ``remember()``, which puts ``project`` and
``session_id`` in the SAME metadata dict as two *different* keys).

The v3 branch of ``GET /api/memories`` instead aliased
``atomic_facts.session_id AS project_name`` — a different field entirely, one
that is only ~5% populated on a real store (see the comment above
``session_id = resolve_session_id(...)`` in ``mcp/tools_core.py``). A memory
saved with a real project therefore showed "-" in the dashboard's Memories ->
All memories table, and ``?project_name=`` filtered on the wrong column.

``retrieval.project_scope.stored_projects()`` already reads the metadata
field correctly for project-aware recall; this test pins the dashboard list
route to the same source.
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
    # m-proj: saved under a project, with an UNRELATED session_id — if the
    # route ever regresses back to reading session_id, this would make the
    # test fail by showing "sess-unrelated" instead of "superlocalmemory".
    engine._db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        " speaker, role, created_at, metadata_json, scope) "
        "VALUES ('m-proj','default','source','sess-unrelated','user','user',"
        " '2026-01-01T00:00:00Z','{\"project\":\"superlocalmemory\"}','personal')"
    )
    # m-noproj: no project in metadata at all (most memories, pre-4.1.21).
    engine._db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        " speaker, role, created_at, metadata_json, scope) "
        "VALUES ('m-noproj','default','source','sess-unrelated','user','user',"
        " '2026-01-01T00:00:00Z','{}','personal')"
    )
    for fid, memory_id, content in (
        ("fact-proj", "m-proj", "The payments ledger will use PostgreSQL 16"),
        ("fact-noproj", "m-noproj", "ledger is a type of cryptocurrency"),
    ):
        engine._db.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
            " lifecycle, created_at, scope, quarantined) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (fid, memory_id, "default", content, "active",
             "2026-01-01T00:00:00Z", "personal", 0),
        )

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    yield TestClient(app), _daemon_headers(app)


def _by_id(payload: dict) -> dict[str, dict]:
    return {m["id"]: m for m in payload.get("memories", [])}


class TestProjectNameReadsSavedMetadataNotSessionId:
    def test_a_memory_saved_with_a_project_shows_it(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50", headers=h)
        assert r.status_code == 200, r.text
        rows = _by_id(r.json())
        assert rows["fact-proj"]["project_name"] == "superlocalmemory", (
            f"expected the saved project, got {rows['fact-proj']['project_name']!r} "
            "(session_id would show 'sess-unrelated' if this regressed)"
        )

    def test_a_memory_saved_with_no_project_is_falsy(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50", headers=h)
        rows = _by_id(r.json())
        assert not rows["fact-noproj"]["project_name"], (
            f"a fact with no project metadata got {rows['fact-noproj']['project_name']!r}"
        )

    def test_filtering_by_project_name_matches_the_metadata_project(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50&project_name=superlocalmemory", headers=h)
        assert r.status_code == 200, r.text
        ids = list(_by_id(r.json()))
        assert ids == ["fact-proj"], (
            f"filtering on the real project returned {ids}"
        )

    def test_filtering_by_an_unrelated_session_id_matches_nothing(self, client) -> None:
        """Pins the old bug: this used to be exactly what the filter matched."""
        tc, h = client
        r = tc.get("/api/memories?limit=50&project_name=sess-unrelated", headers=h)
        assert r.status_code == 200, r.text
        assert list(_by_id(r.json())) == []
