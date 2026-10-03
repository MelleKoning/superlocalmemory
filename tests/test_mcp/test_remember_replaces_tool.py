# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The ``remember`` tool's ``replaces``: refused before any work when malformed,
sent to the daemon when well formed (also through the pool proxy), a daemon
refusal is a clear non-retryable error, and the outcome reaches the caller."""

from __future__ import annotations

import asyncio

import pytest

CONTENT = "The release train leaves on Thursday at 09:00 UTC."
OLD_ID = "3f2a9c0d11e84b7a"
REPLACED = {"ok": True, "replaces": OLD_ID, "fact_ids": [OLD_ID],
            "cases": [{"case_id": "c1", "version": 1}], "undo": "x"}


def _tools() -> dict:
    from superlocalmemory.mcp import tools_core

    captured = {}

    class _Server:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    tools_core.register_core_tools(_Server(), lambda: None)
    return captured


def _daemon(monkeypatch, reply=None, error=None) -> list:
    from superlocalmemory.cli import daemon

    calls = []

    def request(method, path, body=None, **kwargs):
        calls.append((path, body, kwargs))
        if error is not None:
            raise error
        return reply

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", request)
    return calls


@pytest.mark.parametrize("value", ["", "   ", 7, ["abc"], "not an id"])
def test_a_malformed_replaces_is_refused_before_any_work(monkeypatch, value) -> None:
    from superlocalmemory.cli import daemon

    monkeypatch.setattr(daemon, "is_daemon_running",
                        lambda: pytest.fail("contacted the daemon"))
    out = asyncio.run(_tools()["remember"](CONTENT, replaces=value))
    assert out["success"] is False and out["code"] == "INVALID_REPLACES"
    assert out["retryable"] is False and out["error"]


def test_replaces_goes_to_the_daemon_and_the_outcome_comes_back(monkeypatch) -> None:
    calls = _daemon(monkeypatch, reply={"ok": True, "fact_ids": ["f-new"], "count": 1,
                                        "operation_id": "op", "replaced": REPLACED})
    out = asyncio.run(_tools()["remember"](CONTENT, replaces=f" {OLD_ID} "))
    [(path, body, kwargs)] = calls
    assert path == "/remember" and body["replaces"] == OLD_ID
    assert kwargs.get("preserve_unprocessable") is True
    assert out["success"] is True and out["replaced"] == REPLACED


def test_a_daemon_refusal_is_a_clear_error_not_an_outage(monkeypatch) -> None:
    from superlocalmemory.cli.daemon import DaemonUnprocessable

    calls = _daemon(monkeypatch, error=DaemonUnprocessable(
        "REPLACES_NOT_FOUND", "No memory with id x in this profile."))
    out = asyncio.run(_tools()["remember"](CONTENT, replaces=OLD_ID))
    assert out == {"success": False, "code": "REPLACES_NOT_FOUND", "retryable": False,
                   "error": "No memory with id x in this profile."}
    assert len(calls) == 1


def test_a_permission_refusal_is_not_reported_as_an_outage(monkeypatch) -> None:
    """Replacing needs the right to correct; a refusal must not invite retries."""
    from superlocalmemory.cli.daemon import DaemonRefused

    calls = _daemon(monkeypatch, error=DaemonRefused(403, "/remember"))
    out = asyncio.run(_tools()["remember"](CONTENT, replaces=OLD_ID))
    assert out["success"] is False and out["code"] == "NOT_AUTHORIZED"
    assert out["retryable"] is False and len(calls) == 1


def test_the_pool_proxy_reports_a_permission_refusal(monkeypatch) -> None:
    from superlocalmemory.cli.daemon import DaemonRefused
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    _daemon(monkeypatch, error=DaemonRefused(403, "/remember"))
    out = DaemonPoolProxy(port=1).store(CONTENT, {"replaces": OLD_ID})
    assert out["ok"] is False and out["code"] == "NOT_AUTHORIZED"
    assert out["retryable"] is False


def test_without_replaces_the_daemon_request_is_unchanged(monkeypatch) -> None:
    calls = _daemon(monkeypatch, reply={"ok": True, "fact_ids": ["f"], "count": 1})
    out = asyncio.run(_tools()["remember"](CONTENT))
    [(_path, body, kwargs)] = calls
    assert "replaces" not in body and "preserve_unprocessable" not in kwargs
    assert "replaced" not in out


def test_the_pool_proxy_passes_replaces_through(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    calls = _daemon(monkeypatch, reply={"ok": True, "fact_ids": ["f-new"], "count": 1,
                                        "replaced": REPLACED})
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: False)
    monkeypatch.setattr(_daemon_proxy, "choose_pool",
                        lambda: _daemon_proxy.DaemonPoolProxy(port=1))
    out = asyncio.run(_tools()["remember"](CONTENT, replaces=OLD_ID))
    [(path, body, kwargs)] = calls
    assert body["replaces"] == OLD_ID
    assert "replaces" not in body["metadata"]
    assert kwargs.get("preserve_unprocessable") is True
    assert out["success"] is True and out["replaced"] == REPLACED


def test_the_pool_proxy_reports_a_daemon_refusal(monkeypatch) -> None:
    from superlocalmemory.cli.daemon import DaemonUnprocessable
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    _daemon(monkeypatch, error=DaemonUnprocessable("REPLACES_NOT_ALLOWED", "theirs"))
    out = DaemonPoolProxy(port=1).store(CONTENT, {"replaces": OLD_ID})
    assert out == {"ok": False, "code": "REPLACES_NOT_ALLOWED", "retryable": False,
                   "error": "theirs"}


def test_the_pool_proxy_without_replaces_is_unchanged(monkeypatch) -> None:
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    calls = _daemon(monkeypatch, reply={"ok": True, "fact_ids": ["f"]})
    DaemonPoolProxy(port=1).store(CONTENT, {"tags": "a"})
    [(_path, body, kwargs)] = calls
    assert "replaces" not in body and "preserve_unprocessable" not in kwargs
