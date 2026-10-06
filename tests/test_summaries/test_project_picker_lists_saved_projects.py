# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Summaries picker must list a project known only from saved memories.

Issues #113 (saved views + summaries) and #150 (recall knows your project)
both ship in 4.1.21. #150 taught ``remember(project=...)`` to tag a memory
with a project, and ``summaries/project_work_log.py`` already merges those
saved rows into a project's work log (its ``saved_rows`` query, matched by
``core.project_identity.project_key``). But ``GET /api/summary/projects`` —
the picker that feeds the Summaries tab's project dropdown — read ONLY
``tool_events.project_path`` (the directory an agent's hook ran in). A
project with memories saved under it and no hook-observed tool-event session
showed in the Memories page as "N memories across M projects" and in the
table's Project column, but the picker said "No projects recorded yet" and
``GET /api/summary/projects`` returned ``{"projects": []}`` — the picker and
the summary disagreeing about what a project is.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import summaries as routes
from superlocalmemory.storage.schema import create_all_tables
from superlocalmemory.storage.schema_v347 import apply_v347_schema


def _save_memory(conn: sqlite3.Connection, memory_id: str, fact_id: str, content: str,
                  project: str | None, profile: str = "default") -> None:
    import json

    metadata = json.dumps({"project": project}) if project else "{}"
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content, metadata_json) "
        "VALUES (?, ?, ?, ?)", (memory_id, profile, content, metadata))
    conn.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        "importance, created_at) VALUES (?, ?, ?, ?, 0.5, '2026-10-05T10:00:00Z')",
        (fact_id, memory_id, profile, content))


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "get_active_profile", lambda: "default")
    monkeypatch.setattr(routes, "_load_config", lambda: None)
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        create_all_tables(conn)
        apply_v347_schema(conn)  # tool_events, as the engine does
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _db(tmp_path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "memory.db")


class TestPickerListsSavedProjects:
    """The defect: a project saved on a memory, with zero tool_events, was
    invisible to the picker even though the work log could already
    summarise it."""

    def test_a_project_known_only_from_saved_memories_is_listed(self, client, tmp_path):
        with _db(tmp_path) as conn:
            _save_memory(conn, "m1", "f1", "Billing cutover decision.", "acme-billing")
            _save_memory(conn, "m2", "f2", "Billing rollback plan.", "acme-billing")
            conn.commit()

        body = client.get("/api/summary/projects").json()
        paths = {p["path"] for p in body["projects"]}
        assert "acme-billing" in paths, body

        picked = next(p for p in body["projects"] if p["path"] == "acme-billing")
        assert picked["events"] == 2
        assert picked["label"] == "acme-billing"

    def test_picking_it_produces_a_non_empty_summary(self, client, tmp_path):
        """The whole point of listing it: picking the entry must actually
        summarise the memories saved under it, not come back empty."""
        with _db(tmp_path) as conn:
            _save_memory(conn, "m1", "f1", "Billing cutover decision.", "acme-billing")
            _save_memory(conn, "m2", "f2", "Billing rollback plan.", "acme-billing")
            conn.commit()

        projects = client.get("/api/summary/projects").json()["projects"]
        target = next(p["path"] for p in projects if p["path"] == "acme-billing")

        result = client.get(
            "/api/summary", params={"kind": "project", "target": target}).json()
        assert set(result["source_fact_ids"]) == {"f1", "f2"}
        assert "No tool events or facts found" not in result["summary"]
        assert result["coverage"] != "insufficient"

    def test_archived_and_other_profile_memories_are_excluded(self, client, tmp_path):
        with _db(tmp_path) as conn:
            _save_memory(conn, "m1", "f1", "Visible memory.", "acme-billing")
            _save_memory(conn, "m2", "f2", "Archived memory.", "acme-billing")
            conn.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = 'f2'")
            _save_memory(conn, "m3", "f3", "Other profile memory.", "acme-billing",
                         profile="work")
            conn.commit()

        body = client.get("/api/summary/projects").json()
        picked = next(p for p in body["projects"] if p["path"] == "acme-billing")
        assert picked["events"] == 1

    def test_a_project_with_both_a_working_directory_and_saved_memories_is_one_entry(
            self, client, tmp_path):
        """#150's identity rule: the same project named two ways (a tool-event
        working directory, and the name a memory was saved under) must not
        appear twice in the picker."""
        with _db(tmp_path) as conn:
            conn.execute(
                "INSERT INTO tool_events (session_id, profile_id, project_path, "
                "tool_name, event_type, created_at) VALUES "
                "('s1', 'default', '/Users/alice/work/acme-billing', 'Bash', 'call', "
                "'2026-10-05T10:00:00Z')")
            _save_memory(conn, "m1", "f1", "Billing note.", "ACME-Billing")
            conn.commit()

        body = client.get("/api/summary/projects").json()
        matches = [p for p in body["projects"]
                   if p["path"].rstrip("/").lower().endswith("acme-billing")]
        assert len(matches) == 1, body["projects"]

    def test_truncated_still_reflects_only_the_tool_events_ring(self, client, tmp_path):
        """A saved-only project must never be the reason ``truncated`` flips —
        that flag describes the tool_events ring buffer, nothing else."""
        with _db(tmp_path) as conn:
            _save_memory(conn, "m1", "f1", "Note.", "solo-project")
            conn.commit()

        body = client.get("/api/summary/projects").json()
        assert body["truncated"] is False
        assert body["event_rows"] == 0

    def test_no_projects_at_all_still_answers_with_an_empty_list(self, client):
        body = client.get("/api/summary/projects").json()
        assert body["projects"] == []
        assert body["truncated"] is False
