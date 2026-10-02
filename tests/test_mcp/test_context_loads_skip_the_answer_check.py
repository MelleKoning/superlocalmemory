# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Recalls nobody typed as a question are not sent to be judged.

Session start (``session_init``, ``POST /session/open``), the auto-injected
``slm://context`` resource and AutoRecall load context. With the online check
on, judging them would send the project path as "your question", with the
user's top memories, to a provider — and bill the user's key — on every
session start. Each surface marks its recall (``skip_answer_check``); across
HTTP the marker travels as ``answer_check=skip``.

These tests prove each surface SETS the marker or SENDS the parameter. The
recall pipeline honouring the marker is tested where that gate lives.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.answer_check_scope import answer_check_skipped
from superlocalmemory.mcp._pool_adapter import PoolFact, PoolRecallItem, PoolRecallResponse


class _MockServer:
    def __init__(self):
        self.tools: dict[str, object] = {}
        self.resources: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator

    def resource(self, uri, *args, **kwargs):
        def decorator(fn):
            self.resources[uri] = fn
            return fn
        return decorator


def _pool_response() -> PoolRecallResponse:
    return PoolRecallResponse(results=[PoolRecallItem(
        fact=PoolFact(fact_id="f-0", content="JWT with 1h expiry", memory_id="m-0"),
        score=0.9,
    )])


def _recorder(seen: list):
    def fake_pool_recall(query, limit=10, **kwargs):
        seen.append({"query": query, "skipped": answer_check_skipped(), **kwargs})
        return _pool_response()
    return fake_pool_recall


def _engine():
    engine = MagicMock()
    engine.profile_id = "default"
    engine.db.get_pinned.return_value = []
    engine._auto_invoker = None
    return engine


# ---------------------------------------------------------------------------
# session_init — never a question, with or without an explicit query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"project_path": "/Users/someone/private-project"},
    {"query": "SLM release plan"},
    {},
])
def test_session_init_recall_skips_the_answer_check(kwargs):
    from superlocalmemory.mcp.tools_active import register_active_tools

    seen: list = []
    srv = _MockServer()
    register_active_tools(srv, lambda: _engine())
    rules = MagicMock()
    rules.should_recall.return_value = True
    rules.get_recall_config.return_value = {"relevance_threshold": 0.3}
    with patch("superlocalmemory.hooks.rules_engine.RulesEngine", return_value=rules), \
         patch("superlocalmemory.mcp._pool_adapter.pool_recall", _recorder(seen)), \
         patch("superlocalmemory.mcp.tools_active._emit_event", create=True):
        result = asyncio.run(srv.tools["session_init"](**kwargs))
    assert result["success"] is True
    assert seen and all(call["skipped"] for call in seen)


# ---------------------------------------------------------------------------
# slm://context and AutoRecall
# ---------------------------------------------------------------------------

def test_slm_context_resource_recall_skips_the_answer_check():
    from superlocalmemory.mcp.resources import register_resources

    seen: list = []
    srv = _MockServer()
    register_resources(srv, lambda: _engine())
    with patch("superlocalmemory.mcp._pool_adapter.pool_recall", _recorder(seen)):
        text = asyncio.run(srv.resources["slm://context"]())
    assert "JWT with 1h expiry" in text
    assert seen and all(call["skipped"] for call in seen)


def test_auto_recall_engine_path_skips_the_answer_check():
    from superlocalmemory.hooks.auto_recall import AutoRecall

    seen: list = []

    class _Engine:
        _auto_invoker = None

        def recall(self, query, limit=10):
            seen.append(answer_check_skipped())
            return SimpleNamespace(results=[], abstention_reason=None)

    AutoRecall(engine=_Engine()).get_session_context(project_path="/p")
    assert seen == [True]


def test_the_marker_does_not_leak_past_the_context_load():
    from superlocalmemory.hooks.auto_recall import AutoRecall

    AutoRecall(recall_fn=lambda q, limit=10: None).get_session_context(query="x")
    assert answer_check_skipped() is False


# ---------------------------------------------------------------------------
# across HTTP: the daemon proxy sends answer_check=skip
# ---------------------------------------------------------------------------

def _proxy_paths(monkeypatch):
    from superlocalmemory.cli import daemon as daemon_client

    paths: list[str] = []

    def fake_request(method, path, *args, **kwargs):
        paths.append(path)
        return {"ok": True, "results": []}

    monkeypatch.setattr(daemon_client, "daemon_request", fake_request)
    return paths


def test_the_proxy_sends_the_skip_from_a_marked_context(monkeypatch):
    from superlocalmemory.core.answer_check_scope import skip_answer_check
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    paths = _proxy_paths(monkeypatch)
    with skip_answer_check():
        DaemonPoolProxy(port=1).recall("project context /p", limit=5)
    assert "answer_check=skip" in paths[0]


def test_the_proxy_sends_the_skip_when_asked_explicitly(monkeypatch):
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    paths = _proxy_paths(monkeypatch)
    DaemonPoolProxy(port=1).recall("q", limit=5, answer_check=False)
    assert "answer_check=skip" in paths[0]


def test_an_ordinary_recall_query_string_is_unchanged(monkeypatch):
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    paths = _proxy_paths(monkeypatch)
    DaemonPoolProxy(port=1).recall("what did we decide?", limit=5)
    assert "answer_check" not in paths[0]


def test_pool_recall_forwards_an_explicit_skip(monkeypatch):
    from superlocalmemory.mcp import _pool_adapter

    calls: list[dict] = []

    class _Pool:
        def recall(self, **kwargs):
            calls.append(kwargs)
            return {"ok": True, "results": []}

    monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Pool())
    _pool_adapter.pool_recall("q", limit=3, answer_check=False)
    _pool_adapter.pool_recall("q", limit=3)
    assert calls[0].get("answer_check") is False
    assert "answer_check" not in calls[1]


# ---------------------------------------------------------------------------
# the per-prompt hook's direct HTTP fallback
# ---------------------------------------------------------------------------

def test_the_prompt_hook_http_fallback_sends_the_skip(monkeypatch):
    from superlocalmemory.cli import daemon as daemon_client
    from superlocalmemory.hooks import auto_recall_hook

    paths: list[str] = []
    monkeypatch.setattr(daemon_client, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    auto_recall_hook._fallback_recall("fix the login bug", 5, "s-1")
    assert paths and "answer_check=skip" in paths[0]


def test_the_prompt_hook_fallback_only_talks_to_the_verified_daemon(monkeypatch):
    """The prompt the user typed must never go to whatever happens to listen
    on the default port — only to this user's own daemon, checked first."""
    import urllib.request

    from superlocalmemory.cli import daemon as daemon_client
    from superlocalmemory.hooks import auto_recall_hook

    raw_calls: list[str] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"results": [{"content": "injected by a stranger"}]}'

    def raw_http(req, *a, **k):
        raw_calls.append(getattr(req, "full_url", str(req)))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", raw_http)
    monkeypatch.setattr(daemon_client, "daemon_request", lambda *a, **k: None)
    assert auto_recall_hook._fallback_recall("fix the login bug", 5, "s-1") is None
    assert raw_calls == []
