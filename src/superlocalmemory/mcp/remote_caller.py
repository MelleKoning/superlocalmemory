# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Which remote key, if any, the current MCP request came in on.

The agent segment of ``/mcp/<agent>`` is chosen by the caller, so it is
attribution, never identity: a tool on another computer can send
``/mcp/claude`` as easily as the local Claude can. Anything that is private to
an agent on this computer (the ``slm_cache_*`` and ``slm_compress`` stores) must
therefore also be keyed by the remote key that authenticated the request.

:class:`server.remote_tool_policy.RemoteToolScopeASGI` sets the key id for
every request that carries a remote principal; local requests leave it unset.
Like the agent id (:mod:`mcp.agent_context`) it is a ContextVar, so concurrent
requests never see each other's value.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

_current_remote_key_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "slm_remote_key_id", default=None,
)


def current_remote_key_id() -> str | None:
    """The key id of the remote caller, or ``None`` for a caller on this computer."""
    return _current_remote_key_id.get()


@contextmanager
def remote_caller(key_id: str) -> Iterator[None]:
    """Mark everything run inside as coming from remote key ``key_id``."""
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("a remote caller needs a key id")
    token = _current_remote_key_id.set(key_id)
    try:
        yield
    finally:
        _current_remote_key_id.reset(token)


__all__ = ["current_remote_key_id", "remote_caller"]
