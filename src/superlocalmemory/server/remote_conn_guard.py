# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Connection limits for the remote listener.

uvicorn only times a connection out *after* a response (keep-alive). A caller
that completes TLS and then sends nothing, or sends request headers one byte at
a time, holds its connection - and a file handle in the daemon - forever. Many
of them exhaust the daemon's file handles, which takes the local listener and
the database down with it.

On the remote listener only:

* every request's headers must arrive within ``HEADER_DEADLINE_S`` of the
  connection opening (or of the previous response finishing) - a total
  deadline, so trickling bytes does not extend it;
* at most ``MAX_WAITING_CONNECTIONS`` connections may be waiting for request
  headers at once, and at most ``MAX_OPEN_CONNECTIONS`` may be open at all;
  connections over either limit are closed straight away.

A request whose headers have arrived is never cut off by these limits: a long
answer or a streamed reply runs as long as it needs. The TLS handshake itself is
bounded by asyncio (60 seconds) before a connection reaches this code.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("superlocalmemory.remote")

#: Seconds a connection may take to deliver one request's headers.
HEADER_DEADLINE_S = 10.0
#: Connections that may be waiting for request headers at the same time.
MAX_WAITING_CONNECTIONS = 32
#: Connections that may be open on the remote listener at the same time.
MAX_OPEN_CONNECTIONS = 128


class _Limits:
    """Shared by every connection of one remote listener."""

    def __init__(self, deadline_s: float, max_waiting: int, max_open: int) -> None:
        self.deadline_s = deadline_s
        self.max_waiting = max_waiting
        self.max_open = max_open
        self.waiting: set[Any] = set()
        self.open: set[Any] = set()


def guarded_protocol_class(base: type | None = None) -> type:
    """A uvicorn HTTP protocol class that enforces the limits above.

    Read the module limits when called, so each listener gets its own counters.
    """
    if base is None:
        from uvicorn.protocols.http.auto import AutoHTTPProtocol

        base = AutoHTTPProtocol
    limits = _Limits(float(HEADER_DEADLINE_S), int(MAX_WAITING_CONNECTIONS),
                     int(MAX_OPEN_CONNECTIONS))

    class GuardedHTTPProtocol(base):  # type: ignore[misc, valid-type]
        slm_limits = limits

        def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
            self._slm_deadline: asyncio.TimerHandle | None = None
            super().connection_made(transport)
            if (len(limits.open) >= limits.max_open
                    or len(limits.waiting) >= limits.max_waiting):
                logger.warning("Remote listener: too many connections; refusing one.")
                transport.abort()
                return
            limits.open.add(self)
            inner = self.app

            async def app(scope: Any, receive: Any, send: Any) -> Any:
                self._slm_headers_received()
                return await inner(scope, receive, send)

            self.app = app
            self._slm_await_headers()

        def on_response_complete(self) -> None:
            super().on_response_complete()
            if not self.transport.is_closing():
                self._slm_await_headers()

        def connection_lost(self, exc: Exception | None) -> None:
            self._slm_headers_received()
            limits.open.discard(self)
            super().connection_lost(exc)

        def _slm_await_headers(self) -> None:
            self._slm_cancel_deadline()
            limits.waiting.add(self)
            self._slm_deadline = self.loop.call_later(limits.deadline_s, self._slm_expire)

        def _slm_headers_received(self) -> None:
            self._slm_cancel_deadline()
            limits.waiting.discard(self)

        def _slm_cancel_deadline(self) -> None:
            handle = getattr(self, "_slm_deadline", None)
            if handle is not None:
                handle.cancel()
                self._slm_deadline = None

        def _slm_expire(self) -> None:
            self._slm_deadline = None
            limits.waiting.discard(self)
            if not self.transport.is_closing():
                logger.debug("Remote listener: request headers did not arrive in time.")
                self.transport.abort()

    return GuardedHTTPProtocol


__all__ = [
    "HEADER_DEADLINE_S",
    "MAX_OPEN_CONNECTIONS",
    "MAX_WAITING_CONNECTIONS",
    "guarded_protocol_class",
]
