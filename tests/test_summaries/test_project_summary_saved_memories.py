# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""A project summary lists the memories saved under the project (GitHub #150).

It used to read only hook telemetry (tool_events) matched by exact path, and
said "No tool events or facts found" for a project with five memories saved
under it through ``remember(project=...)``.
"""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from superlocalmemory.summaries import generate_project_work_log


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    from superlocalmemory.storage.schema_v347 import apply_v347_schema

    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    apply_v347_schema(str(tmp_path / "memory.db"))  # tool_events, as the engine does
    mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")
    return mgr


def _save(db, content, project=None, profile="default") -> str:
    meta = {"project": project} if project else {}
    memory_id = db.store_memory(MemoryRecord(profile_id=profile, content=content, metadata=meta))
    return db.store_fact(AtomicFact(profile_id=profile, memory_id=memory_id, content=content,
                                    fact_type=FactType.SEMANTIC))


def _five(db) -> list[str]:
    values = ("acme-billing", "/Users/dev/work/acme-billing", "ACME-Billing",
              "/Users/dev/work/acme-billing/", " acme-billing ")
    return [_save(db, f"Billing decision number {i}.", project=v) for i, v in enumerate(values)]


@pytest.mark.parametrize("target", ["acme-billing", "/Users/dev/work/acme-billing",
                                    "ACME-BILLING", "/somewhere/else/acme-billing/"])
def test_summary_lists_memories_saved_under_the_project(db, target) -> None:
    ids = _five(db)
    _save(db, "Unrelated memory.", project="other")
    _save(db, "Untagged memory.")
    _save(db, "Work profile billing note.", project="acme-billing", profile="work")
    result = generate_project_work_log(db.db_path, target, "default")
    assert set(result.source_fact_ids) == set(ids)
    assert "No tool events or facts found" not in result.content
    for i in range(5):
        assert f"Billing decision number {i}." in result.content
    assert "Work profile billing note." not in result.content


def test_tool_events_match_by_name_too(db) -> None:
    db.execute(
        "INSERT INTO tool_events (session_id, profile_id, project_path, tool_name, event_type,"
        " created_at) VALUES ('s1', 'default', '/Users/dev/work/acme-billing/', 'Bash', 'call',"
        " '2026-10-04T10:00:00Z')")
    db.execute(
        "INSERT INTO tool_events (session_id, profile_id, project_path, tool_name, event_type,"
        " created_at) VALUES ('s2', 'default', '/Users/dev/work/acme-billing-v2', 'Edit',"
        " 'call', '2026-10-04T10:00:00Z')")
    result = generate_project_work_log(db.db_path, "acme-billing", "default")
    assert result.metadata["event_count"] == 1
    assert "Bash" in result.content and "Edit" not in result.content


def test_a_project_with_nothing_still_answers_honestly(db) -> None:
    result = generate_project_work_log(db.db_path, "ghost", "default")
    assert result.source_fact_ids == []
    assert "No tool events or facts found" in result.content


def test_the_mcp_tool_lists_them(db, tmp_path, monkeypatch) -> None:
    from superlocalmemory.mcp.tools_summaries import register_summary_tools

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    ids = _five(db)
    tools = {}

    class _Server:
        def tool(self, *a, **k):
            return lambda fn: tools.setdefault(fn.__name__, fn)

    class _Engine:
        profile_id = "default"
        config = None

    register_summary_tools(_Server(), lambda: _Engine())
    out = asyncio.run(tools["get_memory_summary"](kind="project", target="acme-billing"))
    assert out["success"] is True, out
    assert set(out["source_fact_ids"]) == set(ids)
