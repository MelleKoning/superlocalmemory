# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Saving a memory with a kind works the same from HTTP, the tool interface and
the CLI: a known kind is confirmed on the saved memory, an unknown one is
refused with the list of kinds, and nothing is saved by a refused request."""

from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest

from tests.test_server.test_canonical_remember_route import _client

CONTENT = "Never publish a release before the CI matrix is green on every platform."


def _kinds(engine) -> list[tuple]:
    return [tuple(dict(r).values()) for r in engine._db.execute(
        "SELECT memory_kind, memory_kind_source FROM atomic_facts")]


def test_http_remember_with_a_kind_confirms_it(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={"content": CONTENT, "kind": "Rule",
                                                  "idempotency_key": "kind-http-1"})
    assert response.status_code == 200, response.text
    assert _kinds(engine_with_mock_deps) == [("rule", "caller")]


def test_http_remember_with_an_unknown_kind_is_refused(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={"content": CONTENT, "kind": "gossip"})
    assert response.status_code == 422
    assert "rule" in response.text and "decision" in response.text
    assert engine_with_mock_deps._db.execute("SELECT * FROM memories") == []


def test_http_remember_metadata_cannot_smuggle_a_confirmed_kind(engine_with_mock_deps) -> None:
    """L3-13: a caller's own ``metadata`` must not set ``_slm_memory_kind``.

    The documented way to declare a kind is the ``kind`` request field, which
    is parsed and validated before it reaches this metadata slot. Smuggling
    the same key directly through free-form ``metadata`` must not confirm a
    kind at all -- including "rule", which would otherwise create a confirmed
    standing rule through a door neither the memory-kinds HTTP routes, MCP,
    nor the CLI expose.
    """
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={
            "content": CONTENT, "idempotency_key": "kind-smuggle-1",
            "metadata": {"_slm_memory_kind": "rule"},
        })
    assert response.status_code == 200, response.text
    assert _kinds(engine_with_mock_deps) == [(None, None)]


def test_mcp_remember_refuses_an_unknown_kind_before_saving(monkeypatch) -> None:
    from superlocalmemory.mcp import tools_core

    captured = {}

    class _Server:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    tools_core.register_core_tools(_Server(), lambda: None)
    out = asyncio.run(captured["remember"](CONTENT, kind="gossip"))
    assert out["success"] is False and out["code"] == "INVALID_KIND"


def _mcp_remember():
    from superlocalmemory.mcp import tools_core

    captured = {}

    class _Server:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    tools_core.register_core_tools(_Server(), lambda: None)
    return captured["remember"]


def test_mcp_remember_sends_the_kind_as_the_request_field(monkeypatch) -> None:
    """4.1.21 put the kind in metadata, where the daemon strips reserved keys,
    so it never arrived. It must travel as the request's own ``kind`` field.
    The composed path is proven in
    tests/test_integration/test_mcp_declared_kind_transport.py."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.storage.memory_kinds import METADATA_KEY

    sent = {}

    def fake_request(method, path, body=None, **flags):
        sent.update(body=body, flags=flags)
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    asyncio.run(_mcp_remember()(CONTENT, kind="Decision"))
    assert sent["body"]["kind"] == "decision"
    assert METADATA_KEY not in sent["body"]["metadata"]
    assert sent["flags"].get("preserve_unprocessable") is True


def test_mcp_remember_without_a_kind_sends_no_kind_field(monkeypatch) -> None:
    from superlocalmemory.cli import daemon

    sent = {}

    def fake_request(method, path, body=None, **flags):
        sent.update(body=body, flags=flags)
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    asyncio.run(_mcp_remember()(CONTENT))
    assert "kind" not in sent["body"]
    # A 422 (e.g. a reused key) is surfaced with or without a kind (4.1.22).
    assert sent["flags"].get("preserve_unprocessable") is True


def test_mcp_remember_fallback_passes_the_kind_to_the_proxy(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy
    from superlocalmemory.storage.memory_kinds import METADATA_KEY

    sent = {}

    class _Pool:
        def store(self, content, metadata, **kwargs):
            sent.update(metadata=metadata, kwargs=kwargs)
            return {"ok": True, "fact_ids": ["f1"], "count": 1}

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: False)
    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: _Pool())
    asyncio.run(_mcp_remember()(CONTENT, kind="decision"))
    assert sent["kwargs"] == {"kind": "decision"}
    assert METADATA_KEY not in sent["metadata"]


def test_mcp_derived_key_includes_the_kind_only_when_one_is_declared(monkeypatch) -> None:
    from superlocalmemory.cli import daemon

    keys = []

    def fake_request(method, path, body=None, **flags):
        keys.append(body["idempotency_key"])
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    remember = _mcp_remember()
    for kind in ("", "", "rule", "rule", "decision"):
        asyncio.run(remember(CONTENT, kind=kind, agent_id="agent-x"))
    plain, plain_again, rule, rule_again, decision = keys
    assert plain == plain_again and rule == rule_again
    assert len({plain, rule, decision}) == 3


def test_proxy_store_sends_a_declared_kind_and_never_promotes_metadata(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy
    from superlocalmemory.storage.memory_kinds import METADATA_KEY

    sent = []

    def fake_request(method, path, body=None, **flags):
        sent.append((body, flags))
        return {"ok": True, "fact_ids": ["f1"]}

    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    proxy = DaemonPoolProxy(port=1)
    proxy.store(CONTENT, {"idempotency_key": "k1"}, kind="rule")
    proxy.store(CONTENT, {"idempotency_key": "k2", METADATA_KEY: "rule"})
    (declared, declared_flags), (forged, forged_flags) = sent
    assert declared["kind"] == "rule" and declared_flags.get("preserve_unprocessable") is True
    assert "kind" not in forged and forged_flags.get("preserve_unprocessable") is True


def test_cli_remember_refuses_an_unknown_kind_before_contacting_slm(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import commands, daemon

    monkeypatch.setattr(daemon, "daemon_request",
                        lambda *a, **k: pytest.fail("contacted the daemon"))
    with pytest.raises(SystemExit) as exit_info:
        commands.cmd_remember(Namespace(content=CONTENT, tags="", kind="gossip", json=False,
                                        sync_mode=False, scope=None, shared_with=None))
    assert exit_info.value.code == 2
    assert "rule" in capsys.readouterr().err
