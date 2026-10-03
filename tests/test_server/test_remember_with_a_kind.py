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


def test_mcp_remember_sends_the_kind_to_the_daemon(monkeypatch) -> None:
    from superlocalmemory.mcp import _daemon_proxy, tools_core
    from superlocalmemory.storage.memory_kinds import METADATA_KEY

    sent = {}

    class _Pool:
        def store(self, content, metadata):
            sent.update(metadata)
            return {"ok": True, "fact_ids": ["f1"], "count": 1}

    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: _Pool())
    captured = {}

    class _Server:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    tools_core.register_core_tools(_Server(), lambda: None)
    asyncio.run(captured["remember"](CONTENT, kind="decision"))
    assert sent.get(METADATA_KEY) == "decision"


def test_cli_remember_refuses_an_unknown_kind_before_contacting_slm(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import commands, daemon

    monkeypatch.setattr(daemon, "daemon_request",
                        lambda *a, **k: pytest.fail("contacted the daemon"))
    with pytest.raises(SystemExit) as exit_info:
        commands.cmd_remember(Namespace(content=CONTENT, tags="", kind="gossip", json=False,
                                        sync_mode=False, scope=None, shared_with=None))
    assert exit_info.value.code == 2
    assert "rule" in capsys.readouterr().err
