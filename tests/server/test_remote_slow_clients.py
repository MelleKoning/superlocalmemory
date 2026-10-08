# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A caller on the remote listener cannot hold connections open without sending a request.

Each test runs the real remote uvicorn server over TLS on a free loopback port
with small limits, opens at most a handful of connections, and is bounded by
``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import ssl
import time
from pathlib import Path

import pytest
import uvicorn

from superlocalmemory.cli import remote_commands
from superlocalmemory.server import remote_conn_guard, remote_listener

DEADLINE = 0.6


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _app(scope, receive, send):
    await receive()
    if scope["path"].endswith("/slow"):
        await asyncio.sleep(DEADLINE * 2.5)
    body = b"ok"
    if scope["path"].endswith("/scheme"):
        body = b"s1" if scope["scheme"] == "https" else b"s0"
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-length", b"2")]})
    await send({"type": "http.response.body", "body": body})


@pytest.fixture(params=["httptools", "h11"])
def limits(request, monkeypatch):
    """Both uvicorn HTTP implementations: either may be the one installed."""
    from uvicorn.protocols.http import auto
    from uvicorn.protocols.http.h11_impl import H11Protocol

    if request.param == "h11":
        monkeypatch.setattr(auto, "AutoHTTPProtocol", H11Protocol)
    monkeypatch.setattr(remote_conn_guard, "HEADER_DEADLINE_S", DEADLINE)
    monkeypatch.setattr(remote_conn_guard, "MAX_WAITING_CONNECTIONS", 2)
    monkeypatch.setattr(remote_conn_guard, "MAX_OPEN_CONNECTIONS", 3)


def _run(scenario) -> None:
    info = remote_commands.tls_init(["localhost"], ["127.0.0.1"], 30, force=True)
    port = _port()
    cfg = remote_listener.RemoteListenerConfig(
        "127.0.0.1", port, Path(info["server_cert"]), Path(info["server_key"]),
        ("localhost", "127.0.0.1"), remote_listener.datetime.now(remote_listener.timezone.utc))
    main_cfg = uvicorn.Config(_app, host="127.0.0.1", port=_port(), log_level="warning")
    _main, remote_srv = remote_listener.make_servers(_app, main_cfg, cfg)
    sock = remote_listener.bind_socket("127.0.0.1", port)
    tls = ssl.create_default_context(cafile=info["ca"])

    async def connect(*, raw: bool = False):
        if raw:  # TCP only: the caller never starts the TLS handshake
            return await asyncio.open_connection("127.0.0.1", port)
        return await asyncio.open_connection("127.0.0.1", port, ssl=tls,
                                             server_hostname="localhost")

    async def outer():
        task = asyncio.create_task(remote_srv.serve(sockets=[sock]))
        try:
            for _ in range(200):
                if remote_srv.started:
                    break
                await asyncio.sleep(0.02)
            assert remote_srv.started
            await asyncio.wait_for(scenario(connect), 20)
        finally:
            remote_srv.should_exit = True
            await asyncio.wait_for(task, 15)

    try:
        asyncio.run(outer())
    finally:
        sock.close()


async def _closed_within(reader, seconds: float) -> bool:
    try:
        data = await asyncio.wait_for(reader.read(1), seconds)
    except (ConnectionError, ssl.SSLError):
        return True
    except asyncio.TimeoutError:
        return False
    return data == b""


def _close(writer) -> None:
    with contextlib.suppress(Exception):
        writer.close()


def test_a_silent_connection_is_closed_after_the_header_deadline(limits) -> None:
    async def scenario(connect):
        reader, writer = await connect()
        start = time.monotonic()
        assert await _closed_within(reader, DEADLINE + 3), "silent connection kept open"
        assert time.monotonic() - start >= DEADLINE * 0.8
        _close(writer)

    _run(scenario)


def test_trickled_headers_do_not_extend_the_deadline(limits) -> None:
    async def scenario(connect):
        reader, writer = await connect()
        start = time.monotonic()
        closed = False
        for byte in b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Pad: " + b"a" * 200:
            try:
                writer.write(bytes([byte]))
                await writer.drain()
            except (ConnectionError, ssl.SSLError):
                closed = True
                break
            if await _closed_within(reader, 0.05):
                closed = True
                break
            if time.monotonic() - start > DEADLINE + 3:
                break
        assert closed, "trickled headers kept the connection open"
        assert time.monotonic() - start < DEADLINE + 3
        _close(writer)

    _run(scenario)


async def _refused(connect, **kw) -> bool:
    """The listener turned this connection away, at the TLS handshake or just after."""
    try:
        reader, writer = await asyncio.wait_for(connect(**kw), DEADLINE * 0.5)
    except (ConnectionError, ssl.SSLError, OSError, asyncio.IncompleteReadError):
        return True
    except asyncio.TimeoutError:
        return False
    try:
        return await _closed_within(reader, DEADLINE * 0.5)
    finally:
        _close(writer)


def test_connections_waiting_for_headers_are_capped(limits) -> None:
    async def scenario(connect):
        held = [await connect() for _ in range(2)]
        assert await _refused(connect), "third waiting connection accepted"
        for _r, w in held:
            _close(w)

    _run(scenario)


def test_a_connection_that_never_starts_tls_is_closed_after_the_handshake_deadline(
        limits, monkeypatch) -> None:
    """A caller that opens TCP and never sends a TLS hello is cut off, not held 60 s."""
    monkeypatch.setattr(remote_conn_guard, "TLS_HANDSHAKE_TIMEOUT_S", DEADLINE, raising=False)

    async def scenario(connect):
        reader, writer = await connect(raw=True)
        start = time.monotonic()
        assert await _closed_within(reader, DEADLINE + 3), "handshake never bounded"
        assert time.monotonic() - start >= DEADLINE * 0.8
        _close(writer)

    _run(scenario)


def test_connections_still_in_the_tls_handshake_count_toward_the_caps(limits) -> None:
    """Two callers stalled before TLS fill the waiting cap: a third caller is refused."""

    async def scenario(connect):
        stalled = [await connect(raw=True) for _ in range(2)]
        await asyncio.sleep(0.1)  # let the listener accept both
        assert await _refused(connect), "in-handshake connections were not counted"
        for _r, w in stalled:
            _close(w)

    _run(scenario)


def test_the_open_cap_counts_connections_in_the_tls_handshake(limits, monkeypatch) -> None:
    """With the waiting cap out of the way, the open cap still sees stalled callers."""
    monkeypatch.setattr(remote_conn_guard, "MAX_WAITING_CONNECTIONS", 50)

    async def scenario(connect):
        stalled = [await connect(raw=True) for _ in range(3)]
        await asyncio.sleep(0.1)
        assert await _refused(connect), "open cap ignored in-handshake connections"
        for _r, w in stalled:
            _close(w)

    _run(scenario)



def test_requests_on_the_listener_are_seen_as_https(limits) -> None:
    """The remote MCP gate refuses plain HTTP: TLS done by the guard must still say https."""

    async def scenario(connect):
        reader, writer = await connect()
        writer.write(b"GET /mcp/scheme HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        assert head.startswith(b"HTTP/1.1 200"), head
        assert await asyncio.wait_for(reader.readexactly(2), 2) == b"s1"
        _close(writer)

    _run(scenario)


def test_pipelined_requests_are_answered_and_the_counts_come_back(limits, monkeypatch) -> None:
    """Requests sent together right after the handshake are all
    answered, and under the lowest limits a closed connection frees its place
    (handshaking -> open -> gone is counted once each way)."""
    monkeypatch.setattr(remote_conn_guard, "MAX_WAITING_CONNECTIONS", 1)
    monkeypatch.setattr(remote_conn_guard, "MAX_OPEN_CONNECTIONS", 1)

    async def ask_twice(connect) -> None:
        reader, writer = await connect()
        writer.write(b"GET /mcp/a HTTP/1.1\r\nHost: localhost\r\n\r\n"
                     b"GET /mcp/b HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        for _ in range(2):
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            assert head.startswith(b"HTTP/1.1 200"), head
            assert await asyncio.wait_for(reader.readexactly(2), 2) == b"ok"
        return reader, writer

    async def scenario(connect):
        _reader, writer = await ask_twice(connect)
        assert await _refused(connect), "a second connection fit under a cap of one"
        _close(writer)
        await asyncio.sleep(0.2)  # let the listener see the close
        for _ in range(2):  # the place is free again, and stays countable
            _r, w = await ask_twice(connect)
            _close(w)
            await asyncio.sleep(0.2)

    _run(scenario)


def test_a_flood_before_the_hand_over_closes_the_connection(monkeypatch) -> None:
    """Bytes buffered between the handshake and the HTTP hand-over are capped."""
    monkeypatch.setattr(remote_conn_guard, "MAX_PENDING_BYTES", 1000, raising=False)

    class _Transport:
        aborted = False

        def is_closing(self):
            return self.aborted

        def abort(self):
            self.aborted = True

    async def scenario():
        gate_cls = remote_conn_guard.tls_gate_protocol_class(ssl.create_default_context())
        gate = gate_cls(_loop=asyncio.get_running_loop())
        gate._transport = _Transport()
        gate.data_received(b"x" * 600)
        assert not gate._transport.aborted
        gate.data_received(b"x" * 600)
        assert gate._transport.aborted and gate._pending == []

    asyncio.run(scenario())
