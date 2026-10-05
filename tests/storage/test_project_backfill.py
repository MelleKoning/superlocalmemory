# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Session-end summaries get the project their "[name]" prefix names (#150)."""

from __future__ import annotations

import asyncio
import json

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from superlocalmemory.storage.project_backfill import (
    SOURCE_KEY,
    backfill_session_end_projects,
)


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")
    return mgr


def _save(db, content, meta=None, profile="default") -> tuple[str, str]:
    memory_id = db.store_memory(MemoryRecord(profile_id=profile, content=content,
                                             metadata=dict(meta or {})))
    fact_id = db.store_fact(AtomicFact(profile_id=profile, memory_id=memory_id,
                                       content=content, fact_type=FactType.SEMANTIC))
    return memory_id, fact_id


def _meta(db, memory_id) -> dict:
    rows = db.execute("SELECT metadata_json FROM memories WHERE memory_id = ?", (memory_id,))
    return json.loads(dict(rows[0])["metadata_json"])


def test_tags_summaries_and_leaves_everything_else(db) -> None:
    summary, _ = _save(db, "[acme-billing] session ended 2026-10-04 18:20 | branch: main",
                       {"tags": "session-end"})
    work, _ = _save(db, "[platform] session ended 2026-10-03 09:00", profile="work")
    own, _ = _save(db, "[acme-billing] session ended 2026-10-04 19:00",
                   {"project": "/Users/dev/chosen"})
    vague, _ = _save(db, "[acme-billing] session ended at some point")
    plain, _ = _save(db, "A plain memory about invoices.")

    report = backfill_session_end_projects(db.raw_connection)

    assert report.tagged == 2 and report.finished
    assert _meta(db, summary)["project"] == "acme-billing"
    assert _meta(db, summary)[SOURCE_KEY] == "session_end_prefix"
    assert _meta(db, summary)["tags"] == "session-end"
    assert _meta(db, work)["project"] == "platform"            # stays in its profile
    assert _meta(db, own)["project"] == "/Users/dev/chosen"    # never overwritten
    assert "project" not in _meta(db, vague)
    assert "project" not in _meta(db, plain)


def test_is_idempotent_and_batched(db) -> None:
    ids = [_save(db, f"[p{i}] session ended 2026-10-0{1 + i % 9} 10:00")[0] for i in range(7)]
    first = backfill_session_end_projects(db.raw_connection, batch_size=2, max_batches=2)
    assert first.tagged == 4 and not first.finished
    second = backfill_session_end_projects(db.raw_connection, batch_size=2, max_batches=10)
    assert second.tagged == 3 and second.finished
    third = backfill_session_end_projects(db.raw_connection, batch_size=2)
    assert third.tagged == 0 and third.finished
    assert [_meta(db, m)["project"] for m in ids] == [f"p{i}" for i in range(7)]


def test_malformed_metadata_is_left_alone(db) -> None:
    bad, _ = _save(db, "[x] session ended 2026-10-04 10:00")
    db.execute("UPDATE memories SET metadata_json = '{nope' WHERE memory_id = ?", (bad,))
    assert backfill_session_end_projects(db.raw_connection).tagged == 0


def test_fetch_shows_the_backfilled_project(db, tmp_path, monkeypatch) -> None:
    from superlocalmemory.mcp import tools_core

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    _, tagged = _save(db, "[acme-billing] session ended 2026-10-04 18:20")
    _, plain = _save(db, "A plain memory.")
    backfill_session_end_projects(db.raw_connection)
    tools = {}

    class _Server:
        def tool(self, *a, **k):
            return lambda fn: tools.setdefault(fn.__name__, fn)

    class _Engine:
        _db = db
        profile_id = "default"

    tools_core.register_core_tools(_Server(), lambda: _Engine())

    async def _pid(_get):
        return "default"

    monkeypatch.setattr(tools_core, "_runtime_profile", _pid)
    out = asyncio.run(tools["fetch"](f"{tagged},{plain}"))
    projects = {r["fact_id"]: r["project"] for r in out["results"]}
    assert projects == {tagged: "acme-billing", plain: ""}


def test_the_scheduler_runs_it_and_records_the_step(db) -> None:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    summary, _ = _save(db, "[acme-billing] session ended 2026-10-04 18:20")
    sched = MaintenanceScheduler(db, SLMConfig())
    sched._running = True
    sched._project_backfill()
    assert _meta(db, summary)["project"] == "acme-billing"
    assert "project backfill" not in sched.failing_steps()
