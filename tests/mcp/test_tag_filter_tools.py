# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``tags`` / ``tags_match`` on the MCP tools themselves (4.1.22):
``search`` and ``list_recent`` against a real on-disk store, ``recall``
forwarding to the daemon pool and returning ``tag_scope``. Synthetic text."""

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


def _save(db, content, tags=None) -> str:
    memory_id = db.store_memory(MemoryRecord(
        profile_id="default", content=content, metadata={"tags": tags} if tags else {}))
    return db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id,
                                    content=content, fact_type=FactType.SEMANTIC))


def _tools(engine):
    server = _Server()
    tools_core.register_core_tools(server, lambda: engine)
    return server.captured


def test_list_recent_with_a_tag_finds_the_old_memory_and_reports(engine) -> None:
    tagged = _save(engine._db, "We chose the manual gate.", "Decision-Record")
    for i in range(200):
        _save(engine._db, f"status note {i}", "status")
    out = asyncio.run(_tools(engine)["list_recent"](limit=5, tags="decision-record"))
    assert out["success"] is True
    assert [r["fact_id"] for r in out["results"]] == [tagged]
    assert out["tag_scope"]["matched"] == 1 and out["tag_scope"]["applied"] is True


def test_search_with_a_list_of_tags_and_any(engine) -> None:
    a = _save(engine._db, "kestrel alpha", "alpha")
    b = _save(engine._db, "kestrel beta", "beta")
    _save(engine._db, "kestrel gamma", "gamma")
    out = asyncio.run(_tools(engine)["search"](
        "kestrel", limit=10, tags=["alpha", "beta"], tags_match="any"))
    assert out["success"] is True
    assert {r["fact_id"] for r in out["results"]} == {a, b}


def test_search_with_an_unused_tag_says_nobody_saved_it(engine) -> None:
    _save(engine._db, "kestrel alpha", "alpha")
    out = asyncio.run(_tools(engine)["search"]("kestrel", tags="never-used"))
    assert out["results"] == []
    assert out["tag_scope"]["reason"] == "no_memory_has_tag"


def test_without_tags_the_response_is_unchanged(engine) -> None:
    _save(engine._db, "kestrel alpha", "alpha")
    tools = _tools(engine)
    assert "tag_scope" not in asyncio.run(tools["search"]("kestrel"))
    assert "tag_scope" not in asyncio.run(tools["list_recent"](limit=5))


def test_recall_forwards_tags_and_returns_tag_scope(monkeypatch) -> None:
    from superlocalmemory.mcp import _daemon_proxy

    seen: dict = {}
    scope = {"tags": ["decision-record"], "keys": ["decision-record"], "match": "any",
             "applied": True, "matched": 1, "note": ""}

    class _Pool:
        def recall(self, *args, **kwargs):
            seen.update(kwargs)
            return {"ok": True, "results": [{"fact_id": "f1", "content": "x"}],
                    "result_count": 1, "tag_scope": scope if "tags" in kwargs else None}

    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: _Pool())
    server = _Server()
    tools_core.register_core_tools(server, lambda: SimpleNamespace(profile_id="default"))
    out = asyncio.run(server.captured["recall"](
        "q", tags=["decision-record", "a,b"], tags_match="any"))
    assert seen["tags"] == ["decision-record", "a,b"] and seen["tags_match"] == "any"
    assert out["tag_scope"] == scope
    seen.clear()
    asyncio.run(server.captured["recall"]("q"))
    assert "tags" not in seen and "tags_match" not in seen
