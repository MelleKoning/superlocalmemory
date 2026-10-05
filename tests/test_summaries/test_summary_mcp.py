# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``get_memory_summary`` over MCP: session ids an agent can use, this computer's day."""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from superlocalmemory.mcp import tools_summaries
from superlocalmemory.storage.schema import create_all_tables


@pytest.fixture()
def tool(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        create_all_tables(conn)
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                     "VALUES ('m', 'default', 's')")
        conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
                     "session_id, created_at) VALUES ('f1', 'm', 'default', 'note', "
                     "'sess-1', '2026-10-05T20:00:00+00:00')")
    tools: dict = {}

    class Collector:
        def tool(self, *a, **k):
            def register(fn):
                tools[fn.__name__] = fn
                return fn
            return register
    tools_summaries.register_summary_tools(
        Collector(), lambda: SimpleNamespace(profile_id="default", config=None))
    return tools["get_memory_summary"]


def test_a_session_summary_without_an_id_offers_the_ids(tool) -> None:
    answer = asyncio.run(tool(kind="session", target=""))
    assert answer["success"] is False
    assert [s["session_id"] for s in answer["recent_sessions"]] == ["sess-1"]


def test_the_day_uses_this_computers_offset(tool, monkeypatch) -> None:
    from superlocalmemory.summaries import base

    monkeypatch.setattr(base, "local_offset_minutes", lambda day=None: 330)
    answer = asyncio.run(tool(kind="day", target="2026-10-06"))
    assert answer["source_fact_ids"] == ["f1"]
    assert answer["metadata"]["tz_offset_minutes"] == 330


def test_an_unreadable_day_is_refused(tool) -> None:
    answer = asyncio.run(tool(kind="day", target="the fifth"))
    assert answer["success"] is False and "not a date" in answer["error"]
