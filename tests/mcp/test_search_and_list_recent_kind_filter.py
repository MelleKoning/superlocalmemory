# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``search`` and ``list_recent``'s ``kind`` filter on the MCP tool interface
(LLD/WP8 4.1.19). Uses the capture-the-registered-function harness from
test_remember_with_a_kind.py against a real, on-disk DatabaseManager — no
daemon, no embedder.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.mcp import tools_core
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


class _Server:
    def __init__(self) -> None:
        self.captured: dict = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.captured[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def engine(tmp_path):
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    return SimpleNamespace(_db=db, profile_id="default")


def _save(db, content, *, kind=None, source=None, fact_type=FactType.SEMANTIC) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                      fact_type=fact_type, memory_kind=kind, memory_kind_source=source)
    return db.store_fact(fact)


def _tools(engine):
    server = _Server()
    tools_core.register_core_tools(server, lambda: engine)
    return server.captured


# -- list_recent ---------------------------------------------------------------


def test_list_recent_kind_filter_keeps_only_matches(engine) -> None:
    _save(engine._db, "rule one", kind="rule", source="user")
    _save(engine._db, "a decision", kind="decision", source="user")
    tools = _tools(engine)
    out = asyncio.run(tools["list_recent"](limit=10, kind="rule"))
    assert out["success"] is True
    assert [r["content"] for r in out["results"]] == ["rule one"]
    assert out["results"][0]["memory_kind"] == "rule"


def test_list_recent_refuses_an_unknown_kind(engine) -> None:
    tools = _tools(engine)
    out = asyncio.run(tools["list_recent"](limit=10, kind="not-a-real-kind"))
    assert out["success"] is False and out["code"] == "INVALID_KIND"


# -- search ----------------------------------------------------------------


def test_search_kind_filter_keeps_only_matches(engine) -> None:
    _save(engine._db, "alpha fact")
    _save(engine._db, "alpha decision", kind="decision", source="user")
    tools = _tools(engine)
    out = asyncio.run(tools["search"]("alpha", limit=10, kind="decision"))
    assert out["success"] is True
    assert [r["content"] for r in out["results"]] == ["alpha decision"]
    assert out["results"][0]["memory_kind"] == "decision"


def test_search_refuses_an_unknown_kind(engine) -> None:
    tools = _tools(engine)
    out = asyncio.run(tools["search"]("alpha", limit=10, kind="not-a-real-kind"))
    assert out["success"] is False and out["code"] == "INVALID_KIND"
