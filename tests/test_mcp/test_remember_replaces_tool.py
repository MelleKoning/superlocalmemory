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


def _through_the_daemon(monkeypatch, client) -> None:
    """Route the tool's daemon calls into a real daemon app."""
    from superlocalmemory.cli import daemon

    def request(method, path, body=None, **_kwargs):
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        return response.json()

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", request)


def test_same_content_with_a_different_replaces_is_a_different_request(
        engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    engine = engine_with_mock_deps
    remember = _tools()["remember"]
    with _client(engine) as client:
        [first] = client.post("/remember", json={"content": "Plan A: ship on Monday.",
                                                 "idempotency_key": "a"}).json()["fact_ids"]
        [second] = client.post("/remember", json={"content": "Plan B: ship on Tuesday.",
                                                  "idempotency_key": "b"}).json()["fact_ids"]
        _through_the_daemon(monkeypatch, client)
        one = asyncio.run(remember(CONTENT, replaces=first, session_id="s-1"))
        other = asyncio.run(remember(CONTENT, replaces=second, session_id="s-1"))
    assert one["replaced"]["ok"] is True and other["replaced"]["ok"] is True, other
    assert one["operation_id"] != other["operation_id"]


def test_the_same_request_with_the_same_replaces_is_still_one_save(
        engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    engine = engine_with_mock_deps
    remember = _tools()["remember"]
    with _client(engine) as client:
        [old] = client.post("/remember", json={"content": "Plan A: ship on Monday.",
                                               "idempotency_key": "a"}).json()["fact_ids"]
        _through_the_daemon(monkeypatch, client)
        before = len(engine._db.execute("SELECT 1 FROM ingestion_operations"))
        one = asyncio.run(remember(CONTENT, replaces=old, session_id="s-2"))
        again = asyncio.run(remember(CONTENT, replaces=old, session_id="s-2"))
    assert one["operation_id"] == again["operation_id"]
    assert one["replaced"] == again["replaced"] and one["replaced"]["ok"] is True
    assert len(engine._db.execute("SELECT 1 FROM ingestion_operations")) == before + 1
    assert len(engine._db.execute("SELECT 1 FROM correction_cases")) == 1


def test_the_derived_key_follows_replaces(monkeypatch) -> None:
    calls = _daemon(monkeypatch, reply={"ok": True, "fact_ids": ["f"], "count": 1})
    remember = _tools()["remember"]
    for target in (OLD_ID, OLD_ID, "4a1b2c3d4e5f6a7b"):
        asyncio.run(remember(CONTENT, replaces=target, session_id="s-4"))
    keys = [body["idempotency_key"] for _path, body, _kw in calls]
    assert keys[0] == keys[1]
    assert keys[1] != keys[2]


def test_a_plain_call_keeps_its_old_retry_key(monkeypatch) -> None:
    """Without replaces the derived key is what it always was."""
    import hashlib

    calls = _daemon(monkeypatch, reply={"ok": True, "fact_ids": ["f"], "count": 1})
    asyncio.run(_tools()["remember"](CONTENT, agent_id="agent-x", session_id="s-3"))
    material = f"agent-x\0s-3\0\0\0{CONTENT}"
    assert calls[0][1]["idempotency_key"] == "mcp:" + hashlib.sha256(
        material.encode("utf-8")).hexdigest()


def test_the_pool_proxy_without_replaces_is_unchanged(monkeypatch) -> None:
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    calls = _daemon(monkeypatch, reply={"ok": True, "fact_ids": ["f"]})
    DaemonPoolProxy(port=1).store(CONTENT, {"tags": "a"})
    [(_path, body, kwargs)] = calls
    assert "replaces" not in body and "preserve_unprocessable" not in kwargs
