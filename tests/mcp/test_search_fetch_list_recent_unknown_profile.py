# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M1 (4.1.22, Muse 4.1.21 finding): ``search``, ``fetch`` and ``list_recent``
called with an explicit but non-existent ``profile_id`` must refuse, exactly
as ``remember``/``recall`` already do -- not return an empty success.

Before this fix these three tools resolved ``profile_id`` through
``_runtime_profile``, which returns whatever string it is handed unchecked
when non-empty (it only resolves the ACTIVE profile -- via the daemon or the
engine -- when the argument is empty). A typo'd or stale profile name was
silently queried as if it were real, always returning zero results with
``success: true``: indistinguishable from "this profile has nothing
matching," which is a correctness bug on the one signal an agent has for
"did I query the right place."

Uses the same capture-the-registered-function harness as
test_search_and_list_recent_kind_filter.py against a real, on-disk
DatabaseManager -- no daemon, no embedder.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.mcp import tools_core
from superlocalmemory.mcp.request_profile import UNKNOWN_PROFILE
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
    db.initialize(schema)  # seeds the 'default' profile (schema.py)
    return SimpleNamespace(_db=db, profile_id="default")


def _save(db, content: str) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(
        profile_id="default", memory_id=memory_id, content=content,
        fact_type=FactType.SEMANTIC,
    )
    return db.store_fact(fact)


def _tools(engine):
    server = _Server()
    tools_core.register_core_tools(server, lambda: engine)
    return server.captured


# -- search ------------------------------------------------------------------


def test_search_refuses_a_nonexistent_profile_id(engine) -> None:
    _save(engine._db, "alpha fact")
    tools = _tools(engine)

    out = asyncio.run(tools["search"]("alpha", limit=10, profile_id="ghost-profile"))

    assert out["success"] is False
    assert out["code"] == UNKNOWN_PROFILE
    assert out["retryable"] is False
    assert "ghost-profile" in out["error"]


def test_search_still_works_for_the_default_profile(engine) -> None:
    _save(engine._db, "alpha fact")
    tools = _tools(engine)

    out = asyncio.run(tools["search"]("alpha", limit=10, profile_id="default"))

    assert out["success"] is True
    assert [r["content"] for r in out["results"]] == ["alpha fact"]


# -- fetch ---------------------------------------------------------------------


def test_fetch_refuses_a_nonexistent_profile_id(engine) -> None:
    fact_id = _save(engine._db, "alpha fact")
    tools = _tools(engine)

    out = asyncio.run(tools["fetch"](fact_id, profile_id="ghost-profile"))

    assert out["success"] is False
    assert out["code"] == UNKNOWN_PROFILE
    assert "ghost-profile" in out["error"]


def test_fetch_still_works_for_the_default_profile(engine) -> None:
    fact_id = _save(engine._db, "alpha fact")
    tools = _tools(engine)

    out = asyncio.run(tools["fetch"](fact_id, profile_id="default"))

    assert out["success"] is True
    assert out["count"] == 1


# -- list_recent -----------------------------------------------------------------


def test_list_recent_refuses_a_nonexistent_profile_id(engine) -> None:
    _save(engine._db, "alpha fact")
    tools = _tools(engine)

    out = asyncio.run(tools["list_recent"](limit=10, profile_id="ghost-profile"))

    assert out["success"] is False
    assert out["code"] == UNKNOWN_PROFILE
    assert "ghost-profile" in out["error"]


def test_list_recent_still_works_for_the_default_profile(engine) -> None:
    _save(engine._db, "alpha fact")
    tools = _tools(engine)

    out = asyncio.run(tools["list_recent"](limit=10, profile_id="default"))

    assert out["success"] is True
    assert [r["content"] for r in out["results"]] == ["alpha fact"]


def test_list_recent_empty_profile_id_still_resolves_the_active_profile(engine) -> None:
    """Regression guard: an unset profile_id must keep behaving exactly as
    before -- the active profile, never refused."""
    _save(engine._db, "alpha fact")
    tools = _tools(engine)

    out = asyncio.run(tools["list_recent"](limit=10))

    assert out["success"] is True
    assert [r["content"] for r in out["results"]] == ["alpha fact"]
