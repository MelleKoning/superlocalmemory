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


class _Pool:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def recall(self, query, **kw):
        self.calls.append((query, kw))
        return {"ok": True, "profile": kw.get("profile_id"), "no_confident_match": False,
                "results": [{"fact_id": "f-9", "memory_id": "m-9", "content": "nine",
                             "score": 0.9},
                            {"fact_id": "f-3", "memory_id": "m-3", "content": "three",
                             "score": 0.3}]}


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    learning_db(tmp_path)
    profile = {"name": "alice"}

    async def runtime_profile(_get_engine, explicit=""):
        return profile["name"]
    monkeypatch.setattr("superlocalmemory.mcp.tools_core._runtime_profile", runtime_profile)
    pool = _Pool()
    monkeypatch.setattr("superlocalmemory.mcp._daemon_proxy.choose_pool", lambda: pool)
    collector = _Collector()
    tools_views.register_view_tools(collector, lambda: None)
    return collector.tools, pool, profile, ViewStore(tmp_path / "learning.db")


def _call(fn, **kw):
    return asyncio.run(fn(**kw))


def test_create_list_run_rename_delete(mcp) -> None:
    tools, pool, _profile, _store = mcp
    made = _call(tools["manage_view"], action="create", name="Work", query="what shipped",
                 filters={"window": "7d"}, limit=4)
    assert made["success"] and made["view"]["filters"] == {"window": "7d"}
    listed = _call(tools["run_view"])
    assert [v["name"] for v in listed["views"]] == ["Work"]
    run = _call(tools["run_view"], name="work")
    assert run["result_ids"] == ["f-9", "f-3"]
    (query, kw), = pool.calls
    assert query == "what shipped" and kw["limit"] == 4 and kw["window"] == "7d"
    assert kw["profile_id"] == "alice" and kw["session_id"].startswith("view:")
    renamed = _call(tools["manage_view"], action="rename", name="Work", new_name="Shipped")
    assert renamed["view"]["name"] == "Shipped"
    gone = _call(tools["manage_view"], action="delete", name="Shipped")
    assert gone["success"] and _call(tools["run_view"])["views"] == []


def test_a_run_is_not_a_conversation(mcp) -> None:
    """Continuity must ignore view runs, or a second run is biased by the first."""
    from superlocalmemory.core.session_identity import is_conversation

    tools, pool, _profile, store = mcp
    store.create("alice", name="W", query="q")
    _call(tools["run_view"], name="W")
    assert is_conversation(pool.calls[0][1]["session_id"], "alice") is False


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
    tools, pool, profile, store = mcp
    store.create("bob", name="Bob only", query="secret")
    assert _call(tools["run_view"])["views"] == []
    answer = _call(tools["run_view"], name="Bob only")
    assert answer["code"] == "view_not_found" and pool.calls == []
    profile["name"] = "bob"
    assert [v["name"] for v in _call(tools["run_view"])["views"]] == ["Bob only"]


def test_a_failed_recall_is_reported_not_dressed_up(mcp, monkeypatch) -> None:
    tools, _pool, _profile, store = mcp
    store.create("alice", name="W", query="q")

    class Down:
        def recall(self, *a, **k):
            return {"ok": False, "code": "DAEMON_UNAVAILABLE", "retryable": True,
                    "error": "daemon is down"}
    monkeypatch.setattr("superlocalmemory.mcp._daemon_proxy.choose_pool", lambda: Down())
    answer = _call(tools["run_view"], name="W")
    assert answer["success"] is False and answer["code"] == "DAEMON_UNAVAILABLE"


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
