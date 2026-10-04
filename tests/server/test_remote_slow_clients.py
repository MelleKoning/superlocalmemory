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

    async def connect():
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


def test_connections_waiting_for_headers_are_capped(limits) -> None:
    async def scenario(connect):
        held = [await connect() for _ in range(2)]
        reader, writer = await connect()
        assert await _closed_within(reader, DEADLINE * 0.5), "third waiting connection accepted"
        _close(writer)
        for _r, w in held:
            _close(w)

    _run(scenario)


def test_real_requests_are_unaffected(limits) -> None:
    """Keep-alive requests and an answer slower than the deadline still work."""

    async def scenario(connect):
        reader, writer = await connect()
        for path in ("/health", "/mcp/slow", "/health"):
            writer.write(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
            await writer.drain()
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), DEADLINE * 5)
            assert head.startswith(b"HTTP/1.1 200"), head
            assert await asyncio.wait_for(reader.readexactly(2), 2) == b"ok"
        _close(writer)

    _run(scenario)
