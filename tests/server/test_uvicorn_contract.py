# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The uvicorn internals the remote listener relies on, checked by name and by running.

uvicorn has no public API for two servers sharing one application and one
lifespan. ``server/remote_listener.py`` uses five ``uvicorn.Server`` members:
``serve(sockets)``, ``shutdown(sockets)``, ``capture_signals()``,
``should_exit`` and ``started``. If an upgrade changes any of them, a test here
fails with a message that says which, before any user runs it.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
import socket
import ssl
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from superlocalmemory.cli import remote_commands
from superlocalmemory.server import remote_listener

REPO = Path(__file__).resolve().parents[2]


def test_installed_uvicorn_is_the_pinned_and_tested_version() -> None:
    pin = re.search(r'"uvicorn==([0-9.]+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert pin, "pyproject.toml must pin uvicorn exactly (remote access relies on internals)"
    assert uvicorn.__version__ == pin.group(1)
    assert remote_listener.uvicorn_problem() is None
    low, high = remote_listener.UVICORN_TESTED
    major_minor = tuple(int(p) for p in pin.group(1).split(".")[:2])
    assert low <= major_minor < high, "update UVICORN_TESTED together with the pin"


def test_untested_uvicorn_refuses_remote_access(monkeypatch) -> None:
    monkeypatch.setattr(uvicorn, "__version__", "0.99.0")
    assert "not a tested version" in remote_listener.uvicorn_problem()
    remote_commands.tls_init(["localhost"], [], 30, force=True)
    with pytest.raises(remote_listener.RemoteListenerError) as err:
        remote_listener.load_remote_listener_config(env={"SLM_REMOTE_LISTEN": "127.0.0.1:18443"})
    assert err.value.code == "uvicorn_untested"


@pytest.mark.parametrize("member", ["serve", "shutdown", "startup"])
def test_server_coroutines_take_sockets(member) -> None:
    fn = getattr(uvicorn.Server, member, None)
    assert fn is not None and inspect.iscoroutinefunction(fn), (
        f"uvicorn.Server.{member} is no longer a coroutine: remote_listener.serve_pair breaks")
    assert "sockets" in inspect.signature(fn).parameters, (
        f"uvicorn.Server.{member} no longer takes sockets=")


def test_capture_signals_is_a_context_manager_method() -> None:
    fn = getattr(uvicorn.Server, "capture_signals", None)
    assert fn is not None, "uvicorn.Server.capture_signals is gone: QuietServer must change"
    server = uvicorn.Server(uvicorn.Config(lambda *a: None))
    with server.capture_signals():
        pass


def test_should_exit_and_started_are_plain_attributes() -> None:
    server = uvicorn.Server(uvicorn.Config(lambda *a: None))
    assert server.should_exit is False and server.started is False


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_real_pair_serves_both_and_stops_remote_before_the_lifespan_ends() -> None:
    """Two real uvicorn servers, one app, TLS on the remote one."""
    events: list[str] = []

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    events.append("lifespan:start")
                    await send({"type": "lifespan.startup.complete"})
                else:
                    events.append("lifespan:stop")
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        await receive()
        body = scope["path"].encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    info = remote_commands.tls_init(["localhost"], ["127.0.0.1"], 30, force=True)
    main_port, remote_port = _port(), _port()
    cfg = remote_listener.RemoteListenerConfig(
        "127.0.0.1", remote_port, Path(info["server_cert"]), Path(info["server_key"]),
        ("localhost", "127.0.0.1"), remote_listener.datetime.now(remote_listener.timezone.utc))
    main_cfg = uvicorn.Config(app, host="127.0.0.1", port=main_port, log_level="warning")
    main_srv, remote_srv = remote_listener.make_servers(app, main_cfg, cfg)
    main_sock = remote_listener.bind_socket("127.0.0.1", main_port)
    remote_sock = remote_listener.bind_socket("127.0.0.1", remote_port)
    tls = ssl.create_default_context(cafile=info["ca"])
    seen: dict[str, str] = {}

    async def scenario():
        task = asyncio.create_task(remote_listener.serve_pair(
            main_srv, [main_sock], remote_srv, [remote_sock]))
        for _ in range(200):
            if getattr(remote_srv, "started", False):
                break
            await asyncio.sleep(0.05)

        def fetch(url, **kw):
            with urllib.request.urlopen(url, timeout=5, **kw) as resp:
                return resp.read().decode()

        seen["main"] = await asyncio.to_thread(fetch, f"http://127.0.0.1:{main_port}/a")
        seen["remote"] = await asyncio.to_thread(
            fetch, f"https://localhost:{remote_port}/health", context=tls)
        original = remote_srv.shutdown

        async def recording_shutdown(sockets=None):
            events.append("remote:stop")
            await original(sockets=sockets)

        remote_srv.shutdown = recording_shutdown
        main_srv.should_exit = True
        await asyncio.wait_for(task, 30)

    try:
        asyncio.run(scenario())
    finally:
        with contextlib.suppress(OSError):
            main_sock.close()
            remote_sock.close()
    assert seen == {"main": "/a", "remote": "/health"}
    assert events == ["lifespan:start", "remote:stop", "lifespan:stop"], events
