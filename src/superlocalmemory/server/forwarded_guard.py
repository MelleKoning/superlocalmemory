# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Only the socket peer decides whether a caller is on this computer.

SuperLocalMemory trusts a caller on loopback as the local user. Two things
used to let a caller on another computer borrow that trust through a reverse
proxy running on this computer:

* uvicorn rewrote the peer address from ``X-Forwarded-For`` whenever the
  connection came from 127.0.0.1, so a proxy that passed the header through
  turned any claimed address, including ``127.0.0.1``, into the peer;
* a proxy that set no forwarding header at all made every client it carried
  look like 127.0.0.1.

Two rules close both:

1. :func:`uvicorn_proxy_options` turns uvicorn's forwarding-header handling
   off. An operator who runs SLM behind a proxy on purpose names it in
   ``SLM_TRUSTED_PROXIES`` (comma-separated addresses or networks); only then
   are ``X-Forwarded-For`` / ``X-Forwarded-Proto`` read, and only from those
   addresses.
2. :class:`ForwardedLoopbackDemotion` re-addresses any request that arrives
   from loopback carrying a forwarding header to ``forwarded-peer``, which is
   not an address, so no loopback check anywhere in SLM passes for it. A
   request that came through a proxy is never treated as local.

A same-host proxy that strips every forwarding header still looks like
loopback; there is no way to tell it apart from a local program. The
deployment guide says so and recommends SLM's own remote access instead.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from superlocalmemory.server.loopback import is_loopback

#: The scope ``client`` host given to a demoted request. Not an IP address.
FORWARDED_PEER = "forwarded-peer"

#: Request headers that mean "a proxy carried this request for someone else".
FORWARDING_HEADERS = frozenset({
    b"forwarded", b"x-forwarded-for", b"x-real-ip", b"x-forwarded-host",
    b"x-forwarded-proto", b"x-forwarded-port", b"x-forwarded-server",
    b"x-original-forwarded-for", b"true-client-ip", b"cf-connecting-ip",
    b"x-client-ip", b"x-cluster-client-ip", b"fastly-client-ip",
})

TRUSTED_PROXIES_ENV = "SLM_TRUSTED_PROXIES"


def trusted_proxies(env: Mapping[str, str] | None = None) -> str:
    """The operator's ``SLM_TRUSTED_PROXIES`` value, normalised; ``""`` if unset."""
    raw = (env if env is not None else os.environ).get(TRUSTED_PROXIES_ENV, "")
    return ",".join(part.strip() for part in raw.split(",") if part.strip())


def uvicorn_proxy_options(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """uvicorn ``Config`` keyword arguments for forwarding headers.

    Off unless the operator names trusted proxies. uvicorn's own default
    (on, trusting 127.0.0.1, or ``FORWARDED_ALLOW_IPS``) is never used.
    """
    proxies = trusted_proxies(env)
    if not proxies:
        return {"proxy_headers": False, "forwarded_allow_ips": ""}
    return {"proxy_headers": True, "forwarded_allow_ips": proxies}


def has_forwarding_header(headers: Any) -> bool:
    return any(name.lower() in FORWARDING_HEADERS for name, _value in headers or ())


class ForwardedLoopbackDemotion:
    """Pure ASGI. A loopback request that carries a forwarding header is
    re-addressed to :data:`FORWARDED_PEER` before anything else sees it."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in ("http", "websocket"):
            client = scope.get("client") or ("", 0)
            if is_loopback(client[0]) and has_forwarding_header(scope.get("headers")):
                scope = dict(scope, client=(FORWARDED_PEER, client[1]),
                             slm_forwarded_demoted=True)
        await self.app(scope, receive, send)


__all__ = [
    "FORWARDED_PEER",
    "FORWARDING_HEADERS",
    "ForwardedLoopbackDemotion",
    "TRUSTED_PROXIES_ENV",
    "has_forwarding_header",
    "trusted_proxies",
    "uvicorn_proxy_options",
]
