# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Two accounts on one computer: neither talks to, or is mistaken for, the other's daemon.

Each account has its own data folder, descriptor and port files. Every
loopback port is shared, though, so when one account's daemon is down (or on
another port) the default port may be answered by the other account's
SuperLocalMemory. These tests run a real HTTP server on a random loopback port
(never 8765) that answers ``/health`` exactly as another account's daemon
would - its own owner and data folder - and check that nothing of this account
is sent to it and nothing treats it as this account's daemon.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from superlocalmemory.infra import daemon_identity
from superlocalmemory.infra.daemon_identity import (
    DAEMON_PROTOCOL,
    DAEMON_SERVICE,
    build_descriptor,
    namespace_id_for,
    write_descriptor,
)

OTHER_OWNER = "uid:4242424"


class _Daemon(BaseHTTPRequestHandler):
    health: dict = {}
    seen: list[tuple[str, str, dict]] = []

    def _record(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        type(self).seen.append((self.command, self.path, dict(self.headers)))

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        body = json.dumps(type(self).health if self.path == "/health" else {}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    return root


@pytest.fixture
def daemon():
    _Daemon.health = {}
    _Daemon.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Daemon)
    server.daemon_threads = True
    port = int(server.server_address[1])
    assert port != 8765
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, _Daemon
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _other_account_health(port: int, pid: int = 4242) -> dict:
    """What another account's daemon answers: its own owner and data folder."""
    return {
        "status": "ok", "service": DAEMON_SERVICE, "daemon_protocol": DAEMON_PROTOCOL,
        "namespace_id": namespace_id_for("/Users/someone-else/.superlocalmemory"),
        "instance_id": "their-instance", "capability_fingerprint": "f" * 64,
        "owner_id": OTHER_OWNER, "pid": pid, "port": port, "state": "ready",
        "version": "4.1.21",
    }


def _requests_other_than_health(seen) -> list[tuple[str, str]]:
    return [(method, path) for method, path, _h in seen if path != "/health"]


# -- recognising a daemon --------------------------------------------------------------


def test_owned_daemon_answers_only_for_this_accounts_daemon(data_root, daemon) -> None:
    from superlocalmemory.cli.daemon import owned_daemon_answers

    port, server = daemon
    server.health = _other_account_health(port)
    assert not owned_daemon_answers(port)

    mine = build_descriptor(data_root=data_root, port=port, version="4.1.21",
                            pid=os.getpid(), state="ready")
    write_descriptor(mine, data_root=data_root)
    server.health = {"status": "ok", **mine.public_health_fields()}
    assert owned_daemon_answers(port)


def test_a_stale_pid_file_naming_another_accounts_process_is_not_adopted(
        data_root, daemon, monkeypatch) -> None:
    """No descriptor, a leftover daemon.pid/daemon.port, and another account's
    old daemon on that PID and port: it is not this account's daemon."""
    from superlocalmemory.cli import daemon as cli_daemon

    port, server = daemon
    pid = os.getpid()  # a live PID; the process below claims to be the other account's
    (data_root / "daemon.pid").write_text(str(pid), encoding="utf-8")
    (data_root / "daemon.port").write_text(str(port), encoding="utf-8")
    server.health = {"status": "ok", "pid": pid}  # a pre-identity daemon's health

    import psutil

    class _OtherAccountsDaemon:
        def __init__(self, _pid) -> None:
            pass

        def cmdline(self):
            return ["python", "-m", "superlocalmemory.server.unified_daemon", "--start"]

        def uids(self):
            return psutil._common.puids(4242424, 4242424, 4242424)

    monkeypatch.setattr(psutil, "Process", _OtherAccountsDaemon)
    assert cli_daemon._verified_legacy_health() is None
    assert not cli_daemon.is_daemon_running()


def test_a_starting_daemon_is_not_told_another_accounts_daemon_serves_it(daemon) -> None:
    from superlocalmemory.core.remember_runtime import _slm_health_check

    port, server = daemon
    server.health = _other_account_health(port)
    assert not _slm_health_check(port), "another account's daemon taken for this one's"

    sibling = dict(server.health, owner_id=daemon_identity.owner_id(),
                   namespace_id=namespace_id_for(daemon_identity.canonical_data_root()))
    server.health = sibling
    assert _slm_health_check(port), "this account's own sibling daemon must still count"


# -- sending nothing to it ------------------------------------------------------------


def test_the_tool_hook_never_sends_the_install_token_to_another_account(
        data_root, daemon, monkeypatch, capsys) -> None:
    from superlocalmemory.core import security_primitives as sp
    from superlocalmemory.hooks import post_tool_async_hook

    port, server = daemon
    server.health = _other_account_health(port)
    sp.ensure_install_token()
    monkeypatch.setattr(post_tool_async_hook, "DAEMON_URL", f"http://127.0.0.1:{port}")
    payload = json.dumps({"session_id": "s", "tool_name": "Bash",
                          "tool_input": {"command": "cat secrets.txt"},
                          "tool_response": "top secret"})
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(payload))
    assert post_tool_async_hook.main() == 0
    assert _requests_other_than_health(server.seen) == []
    assert not any("x-slm-hook-token" in {k.lower() for k in headers}
                   for _m, _p, headers in server.seen)


def test_slm_ops_refuses_another_accounts_daemon(data_root, daemon, monkeypatch, capsys) -> None:
    from superlocalmemory.cli import ops_cmd

    port, server = daemon
    server.health = _other_account_health(port)
    monkeypatch.setattr(ops_cmd, "_get_daemon_port", lambda: port)
    with pytest.raises(SystemExit) as exit_info:
        ops_cmd._daemon_post("/operations/op-1/resolve", {"action": "cancel"})
    assert exit_info.value.code == 1
    assert _requests_other_than_health(server.seen) == []
    assert "not your SuperLocalMemory" in capsys.readouterr().err


def test_proxy_status_does_not_count_another_accounts_daemon(data_root, daemon) -> None:
    from superlocalmemory.cli import proxy_cmd

    port, server = daemon
    server.health = _other_account_health(port)
    assert not proxy_cmd._ensure_running(port)


def test_start_up_says_the_port_belongs_to_another_account(
        data_root, daemon, monkeypatch, caplog) -> None:
    from superlocalmemory.cli import daemon as cli_daemon

    port, server = daemon
    server.health = _other_account_health(port)
    monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")

    def no_spawn(**_kw):
        raise AssertionError("must not start a daemon onto another account's port")

    monkeypatch.setattr(cli_daemon, "_start_daemon_subprocess", no_spawn)
    with caplog.at_level(logging.ERROR, logger="superlocalmemory.cli.daemon"):
        assert cli_daemon.ensure_daemon(port=port) is False
    assert "another account" in caplog.text
    assert "SLM_DAEMON_PORT" in caplog.text
    assert _requests_other_than_health(server.seen) == []
