# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saved views over MCP: the tools, their refusals, and remote keys (issue #113)."""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.mcp import tools_views
from superlocalmemory.server import remote_profile_binding as binding
from superlocalmemory.server import remote_tool_policy as policy
from superlocalmemory.views import ViewStore

from ..test_security.test_remote_profile_binding import READ_KEY, WRITE_KEY, _run
from ._store import learning_db


class _Collector:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


class _Daemon:
    """The daemon's real views routes, reached the way the tool reaches them.

    ``engine.recall`` is a stand-in that records what it was asked; everything
    between the tool and it is the real code path.
    """

    def __init__(self, profile: dict) -> None:
        from types import SimpleNamespace

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from superlocalmemory.server.routes import views as routes

        self.calls: list[tuple] = []
        self.paths: list[str] = []
        app = FastAPI()
        app.include_router(routes.router)

        def recall(query, **kw):
            self.calls.append((query, kw))
            fact = lambda fid: SimpleNamespace(  # noqa: E731
                fact_id=fid, memory_id="m-" + fid, content=fid, created_at="",
                fact_type=None, lifecycle=None, access_count=0)
            hit = lambda fid, score: SimpleNamespace(  # noqa: E731
                fact=fact(fid), score=score, confidence=0.5, trust_score=0.5,
                channel_scores={}, evidence_chain=[])
            return SimpleNamespace(results=[hit("f-9", 0.9), hit("f-3", 0.3)],
                                   query_type="semantic", channel_weights={},
                                   retrieval_time_ms=1.0, no_confident_match=False)
        app.state.engine = SimpleNamespace(
            recall=recall, _config=SimpleNamespace(), profile_id="alice",
            _db=SimpleNamespace(get_memory_content_batch=lambda *a, **k: {}))
        self.client = TestClient(app)
        self.profile = profile

    def request(self, method, path, body=None, **kw):
        from superlocalmemory.cli.daemon import DaemonNotFound

        self.paths.append(path)
        res = self.client.request(method, path, json=body)
        if res.status_code == 404 and kw.get("preserve_not_found"):
            detail = res.json()["detail"]
            raise DaemonNotFound(404, detail["code"], detail["message"], path)
        return res.json() if res.status_code == 200 else None


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    from superlocalmemory.server.routes import views as routes

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    learning_db(tmp_path)
    profile = {"name": "alice"}

    async def runtime_profile(_get_engine, explicit=""):
        return profile["name"]
    monkeypatch.setattr("superlocalmemory.mcp.tools_core._runtime_profile", runtime_profile)
    monkeypatch.setattr(routes, "_profile", lambda: profile["name"])
    daemon = _Daemon(profile)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", daemon.request)
    collector = _Collector()
    tools_views.register_view_tools(collector, lambda: None)
    return collector.tools, daemon, profile, ViewStore(tmp_path / "learning.db")


def _call(fn, **kw):
    return asyncio.run(fn(**kw))


def test_create_list_run_rename_delete(mcp) -> None:
    tools, daemon, _profile, _store = mcp
    made = _call(tools["manage_view"], action="create", name="Work", query="what shipped",
                 filters={"window": "7d"}, limit=4)
    assert made["success"] and made["view"]["filters"] == {"window": "7d"}
    listed = _call(tools["run_view"])
    assert [v["name"] for v in listed["views"]] == ["Work"]
    run = _call(tools["run_view"], name="work")
    assert run["result_ids"] == ["f-9", "f-3"]
    (query, kw), = daemon.calls
    assert query == "what shipped" and kw["limit"] == 4 and kw["window"] == "7d"
    assert kw["profile_id"] == "alice" and kw["session_id"].startswith("view:")
    # The tool used the daemon's run route, the one run path, labelled "mcp".
    assert daemon.paths == ["/api/v3/views/run?name=work&via=mcp"]
    renamed = _call(tools["manage_view"], action="rename", name="Work", new_name="Shipped")
    assert renamed["view"]["name"] == "Shipped"
    gone = _call(tools["manage_view"], action="delete", name="Shipped")
    assert gone["success"] and _call(tools["run_view"])["views"] == []


def test_a_run_is_not_a_conversation(mcp) -> None:
    """Continuity must ignore view runs, or a second run is biased by the first."""
    from superlocalmemory.core.session_identity import is_conversation

    tools, daemon, _profile, store = mcp
    store.create("alice", name="W", query="q")
    _call(tools["run_view"], name="W")
    assert is_conversation(daemon.calls[0][1]["session_id"], "alice") is False


@pytest.mark.parametrize("kw, code", [
    ({"action": "create", "name": "x" * 81, "query": "q"}, "invalid_view_name"),
    ({"action": "create", "name": "n", "query": "q" * 1001}, "invalid_view_query"),
    ({"action": "create", "name": "n", "query": "q", "limit": 99}, "invalid_view_limit"),
    ({"action": "create", "name": "n", "query": "q", "filters": {"prompt": "x"}},
     "unknown_view_filter"),
    ({"action": "explode", "name": "n"}, "invalid_view_action"),
    ({"action": "delete", "name": "missing"}, "view_not_found"),
])
def test_refusals_carry_stable_codes(mcp, kw, code) -> None:
    tools, *_ = mcp
    answer = _call(tools["manage_view"], **kw)
    assert answer["success"] is False and answer["code"] == code


def test_a_view_of_another_profile_cannot_be_reached(mcp) -> None:
    tools, daemon, profile, store = mcp
    store.create("bob", name="Bob only", query="secret")
    assert _call(tools["run_view"])["views"] == []
    answer = _call(tools["run_view"], name="Bob only")
    assert answer["code"] == "view_not_found" and daemon.calls == []
    profile["name"] = "bob"
    assert [v["name"] for v in _call(tools["run_view"])["views"]] == ["Bob only"]


def test_a_daemon_that_is_down_is_reported_not_dressed_up(mcp, monkeypatch) -> None:
    tools, _daemon, _profile, store = mcp
    store.create("alice", name="W", query="q")
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", lambda *a, **k: None)
    answer = _call(tools["run_view"], name="W")
    assert answer["success"] is False and answer["code"] == "DAEMON_UNAVAILABLE"
    assert answer["retryable"] is True


class TestRemoteKeys:
    def test_a_read_key_may_run_but_not_change_views(self) -> None:
        assert policy.tool_allowed("read", "run_view")
        assert not policy.tool_allowed("read", "manage_view")
        assert policy.tool_allowed("write", "manage_view")
        assert "read-only" in policy.denial_message("manage_view", "viewer", "read")

    def test_views_are_not_routed_so_they_follow_the_keys_profile(self) -> None:
        assert "run_view" not in binding.ROUTED_TOOLS
        assert "manage_view" not in binding.ROUTED_TOOLS

    def test_a_read_key_is_refused_manage_view_at_the_door(self) -> None:
        status, answer, stub, _ = _run("manage_view", {"action": "delete", "name": "x"},
                                       READ_KEY)
        assert answer["result"]["isError"] is True and stub.reached == []

    def test_while_another_profile_is_active_a_key_sees_no_views(self) -> None:
        for principal, tool, args in ((READ_KEY, "run_view", {}),
                                      (READ_KEY, "run_view", {"name": "Work"}),
                                      (WRITE_KEY, "manage_view",
                                       {"action": "create", "name": "x", "query": "q"})):
            _, answer, stub, _ = _run(tool, args, principal, active="secret-client")
            assert answer["result"]["isError"] is True and stub.reached == [], tool
            assert answer["result"]["structuredContent"]["error"] == binding.INACTIVE_DENIAL

    def test_on_its_own_profile_the_call_goes_through_under_a_lease(self) -> None:
        _, answer, stub, _ = _run("run_view", {"name": "Work"}, READ_KEY, active="work")
        assert answer["result"]["isError"] is False and stub.leases_during_call == [1]

    def test_a_profile_hidden_in_filters_is_refused(self) -> None:
        _, answer, stub, _ = _run(
            "manage_view", {"action": "create", "name": "x", "query": "q",
                            "filters": {"profile_id": "secret-client"}}, WRITE_KEY)
        assert answer["result"]["isError"] is True and stub.reached == []
        assert answer["result"]["structuredContent"]["error"] == binding.PROFILE_DENIAL
