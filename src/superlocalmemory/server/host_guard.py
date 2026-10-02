# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Refuse requests addressed to a host name this service is not.

The attack this stops is DNS rebinding. A web page on a domain its owner
controls can point that domain at 127.0.0.1; the browser then treats this
service as part of that site, and the page can read recall results and the
dashboard token as if it were the dashboard. The one thing the browser cannot
disguise is the ``Host`` header: it still names the page's domain. So a
request whose ``Host`` is not this machine is refused before any route runs.

Allowed without configuration:

* ``localhost`` and names under ``.localhost`` (reserved; nobody can own one);
* any IP address — rebinding needs a domain name, so a ``Host`` that is an IP
  literal cannot be one (this keeps LAN access by address working);
* a request with no ``Host`` header at all (browsers always send one).

A deployment reached by a host NAME (``mybox.local``) lists it, comma-separated,
in ``SLM_ALLOWED_HOSTS`` (``*.suffix`` allowed); names already listed in
``SLM_MCP_ALLOWED_HOSTS`` for LAN access count too.
"""

from __future__ import annotations

import ipaddress
import os
import sys
from typing import Any

#: Starlette's test client addresses every request to this name. Accepted only
#: while pytest is loaded — a running SLM service never imports it.
_TEST_CLIENT_HOST = "testserver"

_REFUSAL = (
    b'{"error":"host_not_allowed","message":"This SuperLocalMemory service only '
    b'answers requests addressed to this computer. To reach it by a host name, add '
    b'that name to SLM_ALLOWED_HOSTS."}'
)


def _host_name(raw: str) -> str:
    """The name in a Host header: no port, lower case, no trailing dot."""
    value = raw.strip()
    if value.startswith("["):  # [::1]:8765
        end = value.find("]")
        name = value[1:end] if end > 0 else ""
    elif value.count(":") == 1:  # name:port or v4:port
        name = value.split(":", 1)[0]
    else:  # a bare name, or a bare IPv6 address
        name = value
    return name.strip().rstrip(".").lower()


def _configured_hosts() -> frozenset[str]:
    """Host names the owner allowed: ``SLM_ALLOWED_HOSTS``, plus the names in
    ``SLM_MCP_ALLOWED_HOSTS`` (whose ``name:*`` port patterns LAN setups already
    use), so an existing LAN install keeps working unchanged."""
    names: set[str] = set()
    for var in ("SLM_ALLOWED_HOSTS", "SLM_MCP_ALLOWED_HOSTS"):
        for part in os.environ.get(var, "").split(","):
            entry = part.strip()
            if not entry:
                continue
            if entry != "*" and ":" in entry and not entry.startswith("["):
                entry = entry.rsplit(":", 1)[0]  # name:port or name:*
            names.add(entry.rstrip(".").lower())
    return frozenset(names)


def _matches(name: str, allowed: frozenset[str]) -> bool:
    if "*" in allowed or name in allowed:
        return True
    return any(pattern.startswith("*.") and name.endswith(pattern[1:])
               for pattern in allowed)


def host_allowed(raw: str | None) -> bool:
    """Whether a request carrying this Host header may be served."""
    if raw is None or not raw.strip():
        return True
    name = _host_name(raw)
    if not name:
        return False
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        pass
    if _matches(name, _configured_hosts()):
        return True
    return name == _TEST_CLIENT_HOST and "pytest" in sys.modules


def _header(scope: dict[str, Any], key: bytes) -> str | None:
    for name, value in scope.get("headers") or ():
        if name == key:
            return value.decode("latin-1")
    return None


class HostGuardMiddleware:
    """Pure ASGI: refuses, before anything else runs, a request not addressed
    to this machine (421 for HTTP; policy-violation close for a websocket)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in ("http", "websocket") and not host_allowed(
                _header(scope, b"host")):
            if scope["type"] == "http":
                await send({
                    "type": "http.response.start",
                    "status": 421,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(_REFUSAL)).encode())],
                })
                await send({"type": "http.response.body", "body": _REFUSAL})
            else:
                await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)


__all__ = ["HostGuardMiddleware", "host_allowed"]
