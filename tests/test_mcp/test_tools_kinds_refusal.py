# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L3-04: a daemon refusal (401/403) on the memory-kind MCP tools must map to
NOT_AUTHORIZED, not-retryable — not DAEMON_UNAVAILABLE/retryable=True.

Before the fix, ``_kinds_request`` caught ``DaemonConflict``, ``DaemonNotFound``
and ``DaemonUnprocessable`` explicitly but not ``DaemonRefused`` (a
``RuntimeError`` subclass), so it fell into the bare ``except Exception`` and
was reported as an outage a caller might retry forever, even though the
daemon had already answered and would answer the same way every time.
"""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.cli import daemon as daemon_mod
from superlocalmemory.mcp import tools_kinds


class _Server:
    def __init__(self) -> None:
        self.captured: dict = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.captured[fn.__name__] = fn
            return fn
        return deco


def _tools():
    server = _Server()
    tools_kinds.register_kind_tools(server, lambda: None)
    return server.captured


@pytest.mark.parametrize("tool_name, call", [
    ("set_memory_kind", lambda tools: tools["set_memory_kind"]("f1", "rule")),
    ("confirm_memory_kinds", lambda tools: tools["confirm_memory_kinds"](
        [{"fact_id": "f1", "kind": "rule"}])),
    ("memory_kinds_status", lambda tools: tools["memory_kinds_status"]()),
])
def test_daemon_refused_is_not_authorized_and_not_retryable(
    monkeypatch, tool_name, call,
) -> None:
    monkeypatch.setattr(daemon_mod, "is_daemon_running", lambda: True)

    def _raise_refused(*a, **k):
        raise daemon_mod.DaemonRefused(403, "/api/memory-kinds/status")

    monkeypatch.setattr(daemon_mod, "daemon_request", _raise_refused)

    out = asyncio.run(call(_tools()))
    assert out["success"] is False
    assert out["code"] == "NOT_AUTHORIZED"
    assert out["retryable"] is False
