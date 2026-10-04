# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""prestage_context serves the caller's profile and never moves the engine's.

Before 4.1.20 the tool defaulted to the profile named "default" and assigned
it to the shared engine (never restored), so one call silently moved every
later engine call in the daemon to another profile.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult


class _Engine:
    """Two profiles' memories; recall honours an explicit profile_id."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        self.calls: list[str] = []
        self._facts = {"work": "work: the launch is on tuesday",
                       "default": "default: the dentist is on friday"}

    def recall(self, query, profile_id=None, limit=20, **kwargs) -> RecallResponse:
        pid = profile_id or self.profile_id
        self.calls.append(pid)
        text = self._facts.get(pid)
        results = [] if text is None else [RetrievalResult(
            fact=AtomicFact(fact_id=f"f-{pid}", profile_id=pid, content=text), score=0.9)]
        return RecallResponse(query=query, results=results)


class _Server:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


@pytest.fixture()
def engine(monkeypatch):
    from superlocalmemory.mcp import server as mcp_server

    eng = _Engine("work")
    monkeypatch.setattr(mcp_server, "get_engine", lambda: eng)
    return eng


@pytest.fixture()
def prestage(engine):
    from superlocalmemory.mcp import server as mcp_server
    from superlocalmemory.mcp.tools_context import register_prestage_tool

    srv = _Server()
    register_prestage_tool(srv, mcp_server._prestage_recall)
    return srv.tools["prestage_context"]


async def test_context_comes_from_the_active_profile_and_the_engine_is_not_moved(
        engine, prestage) -> None:
    out = await prestage(query="when is the launch")
    texts = [m["text"] for m in out["memories"]]
    assert texts == ["work: the launch is on tuesday"], out
    assert engine.profile_id == "work"
    assert engine.calls == ["work"]


async def test_an_explicit_profile_is_served_without_moving_the_engine(engine,
                                                                      prestage) -> None:
    out = await prestage(query="dentist", profile_id="default")
    assert [m["text"] for m in out["memories"]] == ["default: the dentist is on friday"]
    assert engine.profile_id == "work"


def test_the_bridge_reads_a_recall_response_and_a_plain_list() -> None:
    from superlocalmemory.mcp.server import _memories_from

    fact = AtomicFact(fact_id="f1", profile_id="work", content="c")
    response = RecallResponse(results=[RetrievalResult(fact=fact, score=0.5)])
    assert _memories_from(response) == [{"id": "f1", "text": "c", "score": 0.5,
                                          "source": "recall"}]
    assert _memories_from([{"fact_id": "f2", "content": "d", "score": 0.1}])[0]["id"] == "f2"
    assert _memories_from(SimpleNamespace(results=None)) == []
