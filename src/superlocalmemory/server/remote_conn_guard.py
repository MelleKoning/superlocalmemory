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
answer or a streamed reply runs as long as it needs.

TLS is set up here too (:func:`tls_gate_protocol_class`), not by uvicorn, so
that a connection is counted from the moment it is accepted:

* the TLS handshake must finish within ``TLS_HANDSHAKE_TIMEOUT_S`` (asyncio's
  own default is 60 seconds);
* a connection still in its handshake counts toward both limits above, and a
  connection over either limit is closed before any TLS work is done.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("superlocalmemory.remote")

#: Seconds a connection may take to deliver one request's headers.
HEADER_DEADLINE_S = 10.0
#: Connections that may be waiting for request headers at the same time.
MAX_WAITING_CONNECTIONS = 32
#: Connections that may be open on the remote listener at the same time.
MAX_OPEN_CONNECTIONS = 128
#: Seconds a connection may take to complete the TLS handshake.
TLS_HANDSHAKE_TIMEOUT_S = 10.0


#: Bytes a connection may send between finishing its TLS handshake and being
#: handed to the HTTP protocol (normally one loop turn). Past this the caller is
#: flooding, not pipelining: the connection is closed instead of buffered.
MAX_PENDING_BYTES = 1 << 20


class _Limits:
    """Shared by every connection of one remote listener.

    ``handshaking`` holds connections whose TLS handshake has not finished; they
    count toward both ``max_waiting`` and ``max_open``.
    """

    def __init__(self, deadline_s: float, max_waiting: int, max_open: int,
                 handshake_s: float = TLS_HANDSHAKE_TIMEOUT_S) -> None:
        self.deadline_s = deadline_s
        self.max_waiting = max_waiting
        self.max_open = max_open
        self.handshake_s = handshake_s
        self.waiting: set[Any] = set()
        self.open: set[Any] = set()
        self.handshaking: set[Any] = set()

    def full(self) -> bool:
        busy = len(self.handshaking)
        return (len(self.open) + busy >= self.max_open
                or len(self.waiting) + busy >= self.max_waiting)


def _read_limits() -> _Limits:
    """The module limits as they are now (tests lower them)."""
    return _Limits(float(HEADER_DEADLINE_S), int(MAX_WAITING_CONNECTIONS),
                   int(MAX_OPEN_CONNECTIONS), float(TLS_HANDSHAKE_TIMEOUT_S))


def guarded_protocol_class(base: type | None = None,
                           limits: _Limits | None = None) -> type:
    """A uvicorn HTTP protocol class that enforces the limits above.

    Read the module limits when called, so each listener gets its own counters.
    """
    if base is None:
        from uvicorn.protocols.http.auto import AutoHTTPProtocol

        base = AutoHTTPProtocol
    if limits is None:
        limits = _read_limits()

    class GuardedHTTPProtocol(base):  # type: ignore[misc, valid-type]
        slm_limits = limits

        def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
            self._slm_deadline: asyncio.TimerHandle | None = None
            super().connection_made(transport)
            if limits.full():
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


def server_tls_context(cert: Path, key: Path) -> ssl.SSLContext:
    """The remote listener's TLS settings: a server context with its certificate."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    return context


def tls_gate_protocol_class(context: ssl.SSLContext, base: type | None = None) -> type:
    """A protocol class for uvicorn's ``http=`` that does TLS itself.

    uvicorn makes one instance per accepted TCP connection (run it *without*
    ``ssl_certfile``). The instance counts the connection at once, refuses it
    when the listener is full, runs the TLS handshake under
    ``TLS_HANDSHAKE_TIMEOUT_S``, then hands the encrypted connection to a
    :func:`guarded_protocol_class` instance built with uvicorn's own arguments.
    """
    limits = _read_limits()
    http_class = guarded_protocol_class(base, limits)

    class TLSGate(asyncio.Protocol):
        slm_limits = limits

        def __init__(self, **http_kwargs: Any) -> None:
            self._make_http: Callable[[], Any] = lambda: http_class(**http_kwargs)
            self._loop = http_kwargs.get("_loop") or asyncio.get_running_loop()
            self._transport: asyncio.Transport | None = None
            self._pending: list[bytes] = []
            self._pending_bytes = 0
            self._eof = False
            self._lost: tuple[Exception | None] | None = None
            self._writing_paused = False
            self._task: asyncio.Task | None = None
            self._http: Any = None

        # -- plain TCP, then the TLS layer until the handshake is done ----------
        def connection_made(self, transport: asyncio.BaseTransport) -> None:  # type: ignore[override]
            self._transport = transport  # type: ignore[assignment]
            if limits.full():
                logger.warning("Remote listener: too many connections; refusing one.")
                transport.abort()  # type: ignore[attr-defined]
                return
            limits.handshaking.add(self)
            # Nothing is read until the TLS layer is in place.
            transport.pause_reading()  # type: ignore[attr-defined]
            self._task = self._loop.create_task(self._handshake())

        def data_received(self, data: bytes) -> None:
            self._pending_bytes += len(data)
            if self._pending_bytes > MAX_PENDING_BYTES:
                logger.warning("Remote listener: too much data before the request "
                               "started; closing the connection.")
                self._pending = []
                self._give_up()
                return
            self._pending.append(data)

        def eof_received(self) -> bool | None:
            self._eof = True
            return None

        def pause_writing(self) -> None:
            self._writing_paused = True

        def resume_writing(self) -> None:
            self._writing_paused = False

        def connection_lost(self, exc: Exception | None) -> None:
            self._lost = (exc,)
            limits.handshaking.discard(self)
            if self._http is not None:
                # Lost between the handshake and the hand-over: the TLS layer
                # had already addressed this call to the gate.
                self._http.connection_lost(exc)

        async def _handshake(self) -> None:
            assert self._transport is not None
            try:
                tls = await self._loop.start_tls(
                    self._transport, self, context, server_side=True,
                    ssl_handshake_timeout=limits.handshake_s)
            except asyncio.CancelledError:
                self._give_up()
                raise
            except OSError as exc:  # includes SSL errors, resets and the timeout
                logger.debug("Remote listener: TLS handshake failed (%s).",
                             type(exc).__name__)
                self._give_up()
                return
            except RuntimeError as exc:  # the transport closed under start_tls
                logger.warning("Remote listener: could not start TLS (%s).", exc)
                self._give_up()
                return
            limits.handshaking.discard(self)
            if tls is None or self._lost is not None:
                self._give_up()
                return
            self._hand_over(tls)

        def _give_up(self) -> None:
            limits.handshaking.discard(self)
            if self._transport is not None and not self._transport.is_closing():
                self._transport.abort()

        def _hand_over(self, tls: asyncio.Transport) -> None:
            """Give the encrypted connection, and anything that already
            arrived on it, to the HTTP protocol."""
            http = self._make_http()
            self._http = http
            tls.set_protocol(http)
            http.connection_made(tls)
            if tls.is_closing():
                return
            if self._writing_paused:
                http.pause_writing()
            for chunk in self._pending:
                http.data_received(chunk)
            self._pending = []
            if self._eof and http.eof_received() is not True:
                tls.close()

        def shutdown(self) -> None:
            """Stop a handshake still in progress (server shutdown)."""
            if self._task is not None and not self._task.done():
                self._task.cancel()

    return TLSGate


__all__ = [
    "HEADER_DEADLINE_S",
    "MAX_OPEN_CONNECTIONS",
    "MAX_PENDING_BYTES",
    "MAX_WAITING_CONNECTIONS",
    "TLS_HANDSHAKE_TIMEOUT_S",
    "guarded_protocol_class",
    "server_tls_context",
    "tls_gate_protocol_class",
]
