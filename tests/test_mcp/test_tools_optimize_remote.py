# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A remote key never reads or writes a local agent's cache (audit 4.1.20 L2 F1).

The ``/mcp/<agent>`` segment is chosen by the caller, so a tool on another
computer could send ``/mcp/claude`` and get the local Claude's cache. Remote
entries are now keyed by the remote key that authenticated the request.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.mcp import agent_context
from superlocalmemory.mcp.remote_caller import current_remote_key_id, remote_caller


class _MockServer:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


@pytest.fixture()
def tools(monkeypatch):
    from superlocalmemory.mcp import tools_optimize

    rows: dict[tuple[str, str], bytes] = {}
    by_key: dict[str, str] = {}

    def _set(cache_key, tenant_id, value_bytes, *, model, ttl_expires, tags):
        rows[(cache_key, tenant_id)] = value_bytes
        by_key[cache_key] = tenant_id

    db = MagicMock()
    db.set.side_effect = _set
    db.get_value.side_effect = lambda cache_key, tenant_id: rows.get((cache_key, tenant_id))
    monkeypatch.setattr(tools_optimize, "CacheDB",
                        type("FakeCDB", (), {"get_default": staticmethod(lambda: db)}))
    server = _MockServer()
    tools_optimize.register_optimize_tools(server)
    server.tools["_by_key"] = by_key
    return server.tools


@pytest.fixture()
def as_agent(monkeypatch):
    """Route as /mcp/<agent>; a fresh ContextVar per test, so nothing leaks."""
    import contextvars

    var: contextvars.ContextVar[str] = contextvars.ContextVar("test_agent", default="")
    monkeypatch.setattr(agent_context, "_current_agent_id", var)
    return var.set


async def test_remote_key_cannot_read_a_local_agents_entry(tools, as_agent) -> None:
    as_agent("claude")
    assert (await tools["slm_cache_set"](key="read:/p/.env", value="TOKEN=local"))["stored"]
    with remote_caller("rk_00000001"):
        got = await tools["slm_cache_get"](key="read:/p/.env")
    assert got["hit"] is False and got["value"] is None


async def test_remote_key_cannot_overwrite_a_local_agents_entry(tools, as_agent) -> None:
    as_agent("claude")
    await tools["slm_cache_set"](key="read:/p/.env", value="TOKEN=local")
    with remote_caller("rk_00000002"):
        planted = await tools["slm_cache_set"](key="read:/p/.env", value="INJECTED")
        own = await tools["slm_cache_get"](key="read:/p/.env")
    assert planted["stored"] is True
    assert own["hit"] is True and own["value"] == "INJECTED"  # its own cache still works
    local = await tools["slm_cache_get"](key="read:/p/.env")
    assert local["value"] == "TOKEN=local"


async def test_two_remote_keys_do_not_share_a_cache(tools, as_agent) -> None:
    as_agent("hermes")
    with remote_caller("rk_aaaaaaaa"):
        await tools["slm_cache_set"](key="k", value="from-a")
    with remote_caller("rk_bbbbbbbb"):
        assert (await tools["slm_cache_get"](key="k"))["hit"] is False
    with remote_caller("rk_aaaaaaaa"):
        assert (await tools["slm_cache_get"](key="k"))["value"] == "from-a"


async def test_remote_cache_key_cannot_collide_with_a_crafted_local_key(tools, as_agent) -> None:
    """Local agent "remote" + key "rk_1:claude:k" must not hash to the remote entry."""
    by_key = tools["_by_key"]
    as_agent("claude")
    with remote_caller("rk_1"):
        await tools["slm_cache_set"](key="k", value="remote")
    remote_keys = set(by_key)
    as_agent("remote")
    await tools["slm_cache_set"](key="rk_1:claude:k", value="local")
    assert len(set(by_key) - remote_keys) == 1


def test_local_entries_keep_their_address_so_existing_caches_still_hit(as_agent) -> None:
    import hashlib

    from superlocalmemory.mcp.tools_optimize import _kv_location
    from superlocalmemory.optimize.storage.db import _normalize_tenant_id

    as_agent("claude")
    assert _kv_location("k") == (hashlib.sha256(b"mcpkv:claude:k").hexdigest(),
                                 _normalize_tenant_id("claude"))


def test_reversible_compression_tenant_is_keyed_by_the_remote_key(as_agent) -> None:
    from superlocalmemory.mcp.tools_optimize import _tenant

    as_agent("claude")
    assert _tenant() == "claude"
    with remote_caller("rk_00000001"):
        assert _tenant() == "remote:rk_00000001:claude"
    assert current_remote_key_id() is None


def test_remote_caller_needs_a_key_id() -> None:
    for bad in ("", None, 5):
        with pytest.raises(ValueError):
            with remote_caller(bad):  # type: ignore[arg-type]
                pass
