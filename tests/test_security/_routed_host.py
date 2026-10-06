# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A computer using profile "personal" that serves a remote key bound to "work".

The real engine, the real daemon routes and the real canonical writer, with
every MCP tool registered against them. ``daemon_request`` is bound to the
daemon's routes, so a tool that goes through the daemon reaches the same route
and writer it would in production. The remote wrapper
(``server/remote_profile_binding``) writes ``profile_id="work"`` into every
call; :meth:`Host.call` does exactly that, then checks the host never moved.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

WORK = "work"
HOST = "personal"
OTHER = "clientx"


class _Server:
    """Collects the functions the ``register_*_tools`` helpers decorate."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorate(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorate


def register_every_tool(get_engine) -> dict[str, Any]:
    from superlocalmemory.mcp import server as mcp_server
    from superlocalmemory.mcp.tools_active import register_active_tools
    from superlocalmemory.mcp.tools_brain import register_brain_tools
    from superlocalmemory.mcp.tools_context import register_prestage_tool
    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.mcp.tools_evolution import register_evolution_tools
    from superlocalmemory.mcp.tools_kinds import register_kind_tools
    from superlocalmemory.mcp.tools_learning import register_learning_tools
    from superlocalmemory.mcp.tools_loops import register_loop_tools
    from superlocalmemory.mcp.tools_summaries import register_summary_tools
    from superlocalmemory.mcp.tools_v3 import register_v3_tools
    from superlocalmemory.mcp.tools_v28 import register_v28_tools
    from superlocalmemory.mcp.tools_v33 import register_v33_tools
    from superlocalmemory.mcp.tools_views import register_view_tools

    server = _Server()
    for register in (register_core_tools, register_active_tools, register_brain_tools,
                     register_evolution_tools, register_kind_tools, register_learning_tools,
                     register_loop_tools, register_summary_tools, register_v3_tools,
                     register_v28_tools, register_v33_tools, register_view_tools):
        register(server, get_engine)
    register_prestage_tool(server, mcp_server._prestage_recall)
    return server.tools


def bind_daemon_request(monkeypatch, client) -> None:
    """``cli.daemon.daemon_request`` answered by the daemon's routes, with the
    same exceptions for the same answers as the real client."""
    from superlocalmemory.cli import daemon

    def request(method, path, body=None, *, preserve_conflict=False,
                preserve_not_found=False, preserve_unprocessable=False, **_kw):
        response = client.request(method, path, json=body)
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = None
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if status in (401, 403):
            raise daemon.DaemonRefused(status, path)
        if status == 409 and preserve_conflict:
            raise daemon.DaemonConflict(str(detail or ""))
        if status == 404 and preserve_not_found:
            # The real client's own reading of a 404 body.
            raise daemon.not_found_from(payload, path)
        if status == 422 and preserve_unprocessable:
            raise daemon.DaemonUnprocessable("INVALID_REQUEST", json.dumps(detail))
        if status >= 400:
            return None
        return payload

    monkeypatch.setattr(daemon, "daemon_request", request)
    monkeypatch.setattr(daemon, "is_daemon_running", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "ensure_daemon", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "_get_port", lambda *a, **k: 1)


@dataclass
class Host:
    engine: Any
    client: Any
    app: Any
    tools: dict[str, Any]

    def call(self, tool: str, /, **arguments: Any) -> dict:
        """Run ``tool`` as the remote wrapper would for a key bound to "work"."""
        before = self.client.get("/status").json()
        result = asyncio.run(self.tools[tool](**arguments, profile_id=WORK))
        after = self.client.get("/status").json()
        assert self.engine.profile_id == HOST, tool
        assert (after["profile"], after.get("profile_generation")) == (
            HOST, before.get("profile_generation")), tool
        return result

    def unrouted(self, tool: str, /, **arguments: Any) -> dict:
        return asyncio.run(self.tools[tool](**arguments))

    def save(self, content: str, profile: str, key: str, **extra: Any) -> str:
        """One memory saved through the daemon into ``profile``; its fact id."""
        response = self.client.post("/remember", json={
            "content": content, "profile_id": profile, "idempotency_key": key, **extra})
        assert response.status_code == 200, response.text
        [fact_id] = response.json()["fact_ids"][:1]
        return fact_id

    def rows(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.engine._db.execute(sql, params)]


def open_host(tmp_path, monkeypatch, mock_embedder):
    """A context manager: the host on "personal", profiles "personal" and "work"."""
    from contextlib import contextmanager

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.mcp import server as mcp_server
    from superlocalmemory.storage.models import Mode
    from tests.test_server.test_per_request_profile import _daemon

    @contextmanager
    def _open():
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        config.retrieval.use_cross_encoder = False
        config.active_profile = HOST
        engine = MemoryEngine(config)
        with patch("superlocalmemory.core.engine_wiring.init_embedder",
                   return_value=mock_embedder):
            engine.initialize()
            engine._embedder = mock_embedder
        assert engine.profile_id == HOST
        try:
            with _daemon(engine, profiles=(HOST, WORK)) as (client, app):
                bind_daemon_request(monkeypatch, client)
                monkeypatch.setattr(mcp_server, "get_engine", lambda: engine)
                yield Host(engine, client, app, register_every_tool(lambda: engine))
        finally:
            engine.close()

    return _open()


__all__ = ["HOST", "OTHER", "WORK", "Host", "open_host", "register_every_tool"]
