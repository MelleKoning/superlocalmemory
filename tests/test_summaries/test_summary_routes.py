# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The summary routes: the caller's day, the session picker, plain refusals (#113)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import summaries as routes
from superlocalmemory.storage.schema import create_all_tables


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "get_active_profile", lambda: "default")
    monkeypatch.setattr(routes, "_load_config", lambda: None)
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        create_all_tables(conn)
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                     "VALUES ('m', 'default', 's')")
        for fid, created, session in (
                ("ist-5th", "2026-10-05T18:00:00+00:00", "s-old"),
                ("ist-6th-a", "2026-10-05T20:00:00+00:00", "s-new"),
                ("ist-6th-b", "2026-10-05T20:30:00+00:00", "s-new")):
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
                "session_id, importance, created_at) VALUES (?, 'm', 'default', ?, ?, "
                "0.5, ?)", (fid, f"note {fid}", session, created))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_the_browser_offset_decides_the_day(client) -> None:
    body = client.get("/api/summary", params={
        "kind": "day", "target": "2026-10-06", "tz_offset_minutes": 330}).json()
    assert sorted(body["source_fact_ids"]) == ["ist-6th-a", "ist-6th-b"]
    assert body["metadata"]["tz_offset_minutes"] == 330
    utc = client.get("/api/summary", params={"kind": "day", "target": "2026-10-06"}).json()
    assert utc["source_fact_ids"] == []


def test_today_is_the_callers_today(client) -> None:
    body = client.get("/api/summary", params={
        "kind": "day", "target": "today", "tz_offset_minutes": 840}).json()
    expected = (datetime.now(UTC) + timedelta(minutes=840)).date().isoformat()
    assert body["metadata"]["date"] == expected


@pytest.mark.parametrize("params", [{"tz_offset_minutes": 841}, {"tz_offset_minutes": "x"},
                                    {"target": "the 5th"}])
def test_unreadable_input_is_a_422(client, params) -> None:
    assert client.get("/api/summary", params={"kind": "day", **params}).status_code == 422


def test_sessions_picker(client) -> None:
    body = client.get("/api/summary/sessions").json()
    assert [s["session_id"] for s in body["sessions"]] == ["s-new", "s-old"]
    assert body["sessions"][0]["memory_count"] == 2
    assert client.get("/api/summary/sessions", params={"limit": 0}).status_code == 422
    one = client.get("/api/summary", params={"kind": "session", "target": "s-new"}).json()
    assert sorted(one["source_fact_ids"]) == ["ist-6th-a", "ist-6th-b"]
