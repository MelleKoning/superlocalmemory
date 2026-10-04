# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The remote listener: configuration refusals, host names, and the server pair."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from superlocalmemory.cli import remote_commands
from superlocalmemory.server import remote_listener
from superlocalmemory.server.remote_listener import RemoteListenerError


def _tls(names=("localhost",), ips=("127.0.0.1",)) -> dict:
    return remote_commands.tls_init(list(names), list(ips), 30, force=True)


def _enable(listen="127.0.0.1:18443") -> None:
    remote_commands.enable(listen)


def test_off_by_default() -> None:
    assert remote_listener.load_remote_listener_config(env={}) is None


def test_valid_config_reads_cert_names() -> None:
    _tls(names=("slm.lan", "localhost"), ips=("192.168.50.144", "::1"))
    _enable()
    cfg = remote_listener.load_remote_listener_config(env={}, main_port=8765)
    assert cfg.port == 18443 and cfg.host == "127.0.0.1"
    assert set(cfg.server_names) == {"slm.lan", "localhost", "192.168.50.144", "[::1]"}
    hosts = remote_listener.mcp_allowed_hosts(cfg)
    assert "slm.lan:18443" in hosts and "[::1]:18443" in hosts


def test_env_overrides_the_saved_setting() -> None:
    _tls()
    _enable("127.0.0.1:18443")
    cfg = remote_listener.load_remote_listener_config(
        env={"SLM_REMOTE_LISTEN": "127.0.0.1:19443"})
    assert cfg.port == 19443


def _write_cert(path: Path, *, expired=False, future=False, san=True, eku=True) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    start = now + timedelta(days=2) if future else now - timedelta(days=10)
    end = now - timedelta(days=1) if expired else now + timedelta(days=10)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    builder = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
               .public_key(key.public_key()).serial_number(1)
               .not_valid_before(start).not_valid_after(end))
    if san:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
    if eku:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    cert = builder.sign(key, hashes.SHA256())
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


@pytest.mark.parametrize("case,code", [
    ("missing_cert", "tls_missing"),
    ("key_0644", "tls_key_exposed"),
    ("expired", "cert_expired"),
    ("not_yet_valid", "cert_not_yet_valid"),
    ("no_san", "cert_no_san"),
    ("no_eku", "cert_no_server_auth"),
    ("stateful", "stateful_mcp"),
    ("same_port", "port_conflict"),
    ("bad_listen", "invalid_listen"),
    ("name_listen", "invalid_listen"),
])
def test_config_refusals(case, code, monkeypatch) -> None:
    info = _tls()
    cert, key = Path(info["server_cert"]), Path(info["server_key"])
    listen = "127.0.0.1:18443"
    if case == "missing_cert":
        cert.unlink()
    elif case == "key_0644":
        if os.name != "posix":
            pytest.skip("POSIX permissions")
        os.chmod(key, 0o644)
    elif case in ("expired", "not_yet_valid", "no_san", "no_eku"):
        _write_cert(cert, expired=case == "expired", future=case == "not_yet_valid",
                    san=case != "no_san", eku=case != "no_eku")
    elif case == "stateful":
        monkeypatch.setenv("SLM_MCP_STATEFUL", "1")
    elif case == "bad_listen":
        listen = "127.0.0.1:99999"
    elif case == "name_listen":
        listen = "slm.lan:8443"
    with pytest.raises(RemoteListenerError) as err:
        remote_listener.load_remote_listener_config(
            env={"SLM_REMOTE_LISTEN": listen}, main_port=18443 if case == "same_port" else 8765)
    assert err.value.code == code


def test_mcp_allowed_hosts_include_cert_sans_in_the_daemon(monkeypatch) -> None:
    from superlocalmemory.server.unified_daemon import _configure_mcp_transport_settings

    _tls(names=("slm.lan",), ips=("192.168.50.144",))
    _enable("127.0.0.1:18443")
    kwargs = _configure_mcp_transport_settings()
    allowed = kwargs["transport_security"].allowed_hosts
    assert "slm.lan:18443" in allowed and "192.168.50.144:18443" in allowed
    assert "127.0.0.1:*" in allowed and "localhost:*" in allowed


def test_remote_listener_host_names_pass_the_host_guard() -> None:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from superlocalmemory.server.host_guard import HostGuardMiddleware

    async def health(_request):
        return JSONResponse({"ok": True})

    inner = HostGuardMiddleware(Starlette(routes=[Route("/health", health)]))
    wrapped = remote_listener.RemoteListenerASGI(inner, ("slm.lan",))
    ok = TestClient(wrapped, base_url="https://slm.lan:8443").get("/health")
    assert ok.status_code == 200
    evil = TestClient(wrapped, base_url="https://evil.example:8443").get("/health")
    assert evil.status_code == 421


# -- the server pair --------------------------------------------------------------------


class _FakeServer:
    def __init__(self, name: str, log: list[str], fail: bool = False) -> None:
        self.name, self.log, self.fail = name, log, fail
        self.started = False
        self.should_exit = False

    async def serve(self, sockets=None) -> None:
        self.log.append(f"{self.name}:start")
        if self.fail:
            raise RuntimeError("remote boom")
        self.started = True
        while not self.should_exit:
            await asyncio.sleep(0.01)
        await self.shutdown(sockets)

    async def shutdown(self, sockets=None) -> None:
        self.log.append(f"{self.name}:stop")


def test_pair_starts_remote_after_main_and_stops_it_when_main_exits() -> None:
    log: list[str] = []
    main, remote = _FakeServer("main", log), _FakeServer("remote", log)

    async def scenario():
        task = asyncio.create_task(remote_listener.serve_pair(main, [], remote, []))
        while not remote.started:
            await asyncio.sleep(0.01)
        main.should_exit = True
        await task

    asyncio.run(scenario())
    assert log.index("main:start") < log.index("remote:start")
    assert "remote:stop" in log


def test_primary_shutdown_stops_remote_before_lifespan_shutdown() -> None:
    import uvicorn

    order: list[str] = []

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    order.append("lifespan:shutdown")
                    await send({"type": "lifespan.shutdown.complete"})
                    return

    cfg = remote_listener.RemoteListenerConfig("127.0.0.1", 0, Path("c"), Path("k"),
                                               ("localhost",), datetime.now(timezone.utc))
    main_cfg = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    main_srv, remote_srv = remote_listener.make_servers(app, main_cfg, cfg)

    class _Remote:
        should_exit = False

    fake_remote = _Remote()

    async def remote_task():
        while not fake_remote.should_exit:
            await asyncio.sleep(0.01)
        order.append("remote:stopped")

    async def scenario():
        main_cfg.load()
        main_srv.lifespan = main_cfg.lifespan_class(main_cfg)
        await main_srv.lifespan.startup()
        main_srv.servers = []
        main_srv._slm_remote_task = asyncio.create_task(remote_task())
        # make_servers' shutdown flips the real remote server; mirror it here.
        original = remote_srv

        async def flip():
            while not original.should_exit:
                await asyncio.sleep(0.01)
            fake_remote.should_exit = True

        asyncio.create_task(flip())
        await main_srv.shutdown(sockets=[])

    asyncio.run(scenario())
    assert order == ["remote:stopped", "lifespan:shutdown"]


def test_a_failing_remote_server_never_stops_the_main_server(caplog) -> None:
    log: list[str] = []
    main, remote = _FakeServer("main", log), _FakeServer("remote", log, fail=True)

    async def scenario():
        task = asyncio.create_task(remote_listener.serve_pair(main, [], remote, []))
        for _ in range(100):
            if "remote:start" in log:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert not task.done()
        main.should_exit = True
        await task

    asyncio.run(scenario())
    assert "main:stop" in log
    assert "Remote listener stopped unexpectedly" in caplog.text


def test_main_listener_starts_when_remote_config_is_broken(monkeypatch, caplog) -> None:
    """start_server logs the refusal and runs the byte-identical single-server path."""
    from superlocalmemory.server import unified_daemon

    _enable("127.0.0.1:18443")  # enabled, but no certificate exists
    monkeypatch.setenv("SLM_DAEMON_PORT", os.environ.get("SLM_DAEMON_PORT", "18765"))
    ran: dict[str, object] = {}

    class _Server:
        def __init__(self, config):
            ran["config"] = config

        def run(self, sockets=None):
            ran["sockets"] = sockets

    import uvicorn

    monkeypatch.setattr(uvicorn, "Server", _Server)
    for name in ("_publish_process_descriptor", "_start_memory_watchdog",
                 "_start_pending_materializer", "_cleanup_process_descriptor",
                 "install_thread_dump_signal", "assert_no_durable_root_conflict"):
        monkeypatch.setattr(unified_daemon, name, lambda *a, **k: None)
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    unified_daemon.start_server(port=port)
    assert ran.get("sockets"), "the main listener did not start"
    assert ran["config"].factory is True
    assert "Remote access is configured but not started" in caplog.text
