# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A request carried by a reverse proxy never gets loopback trust.

The attack: SLM trusts an uncredentialed caller on 127.0.0.1 as the local
user. An operator puts a TLS proxy on the same machine in front of SLM. Before
4.1.20, uvicorn rewrote the peer from ``X-Forwarded-For`` for any connection
from 127.0.0.1, so a remote caller who sent ``X-Forwarded-For: 127.0.0.1``
through a pass-through proxy became "local"; a proxy that set no such header
made every remote caller local.

The live tests run the real daemon app under a real uvicorn server on
127.0.0.1, built with the same options ``start_server`` uses, and make the
request a proxy would make.
"""

from __future__ import annotations

import ast
import http.client
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from starlette.testclient import TestClient

from superlocalmemory.server import forwarded_guard
from superlocalmemory.server.unified_daemon import create_app

_SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"
_WRITE = ("/remember", b'{"content": "planted by a remote caller"}')
_REFUSED = 403  # require_http_mutation_actor's answer for an untrusted writer


def _is_identity_refusal(status: int, body: bytes) -> bool:
    """Refused as a caller from another computer: by the access gate (no
    credentials, 401) or, behind it, by the write-identity check (403)."""
    return ((status == 401 and b"remote_auth_required" in body)
            or (status == _REFUSED and b"Mutation rejected" in body))


# -- in-process: every forwarding header demotes ---------------------------------


@pytest.fixture(scope="module")
def app():
    return create_app()


def _loopback_client(app) -> TestClient:
    return TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8765")


@pytest.mark.parametrize("header,value", [
    ("X-Forwarded-For", "127.0.0.1"),
    ("X-Forwarded-For", "203.0.113.9"),
    ("Forwarded", "for=127.0.0.1"),
    ("X-Real-IP", "127.0.0.1"),
    ("X-Forwarded-Host", "slm.example.com"),
    ("X-Forwarded-Proto", "https"),
])
def test_a_forwarded_loopback_request_cannot_write_uncredentialed(app, header, value) -> None:
    resp = _loopback_client(app).post(_WRITE[0], content=_WRITE[1],
                                      headers={"Content-Type": "application/json",
                                               header: value})
    assert _is_identity_refusal(resp.status_code, resp.content), (resp.status_code, resp.text)


def test_plain_loopback_without_forwarding_headers_is_unchanged(app) -> None:
    resp = _loopback_client(app).post(_WRITE[0], content=_WRITE[1],
                                      headers={"Content-Type": "application/json"})
    assert not _is_identity_refusal(resp.status_code, resp.content), resp.text


# -- uvicorn options -----------------------------------------------------------


def test_forwarding_headers_are_off_unless_the_operator_names_a_proxy(monkeypatch) -> None:
    monkeypatch.delenv(forwarded_guard.TRUSTED_PROXIES_ENV, raising=False)
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")  # uvicorn's own env is ignored
    assert forwarded_guard.uvicorn_proxy_options() == {
        "proxy_headers": False, "forwarded_allow_ips": ""}
    monkeypatch.setenv(forwarded_guard.TRUSTED_PROXIES_ENV, " 10.0.0.5 , 10.1.0.0/16 ")
    assert forwarded_guard.uvicorn_proxy_options() == {
        "proxy_headers": True, "forwarded_allow_ips": "10.0.0.5,10.1.0.0/16"}


def _uvicorn_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "uvicorn"
            and node.func.attr in ("run", "Config")]


def test_every_uvicorn_server_in_slm_takes_the_proxy_options() -> None:
    calls = {p: _uvicorn_calls(p) for p in _SRC.rglob("*.py")}
    calls = {p: c for p, c in calls.items() if c}
    assert calls, "the scan found no uvicorn servers at all"
    for path, found in calls.items():
        for call in found:
            spread = [kw.value for kw in call.keywords if kw.arg is None]
            assert any(isinstance(v, ast.Call) and getattr(v.func, "id", "")
                       == "uvicorn_proxy_options" for v in spread), (
                f"{path.relative_to(_SRC)}:{call.lineno} starts uvicorn without "
                "**uvicorn_proxy_options()")


# -- live: a real uvicorn server, a real socket ------------------------------------


def _serve(app, monkeypatch, trusted: str | None) -> Iterator[int]:
    if trusted is None:
        monkeypatch.delenv(forwarded_guard.TRUSTED_PROXIES_ENV, raising=False)
    else:
        monkeypatch.setenv(forwarded_guard.TRUSTED_PROXIES_ENV, trusted)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off",
                            log_level="warning", **forwarded_guard.uvicorn_proxy_options())
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.fixture()
def live(app, monkeypatch) -> Iterator[int]:
    yield from _serve(app, monkeypatch, trusted=None)


@pytest.fixture()
def live_behind_named_proxy(app, monkeypatch) -> Iterator[int]:
    yield from _serve(app, monkeypatch, trusted="127.0.0.1")


def _post(port: int, headers: dict[str, str]) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("POST", _WRITE[0], body=_WRITE[1],
                     headers={"Content-Type": "application/json",
                              "Host": f"127.0.0.1:{port}", **headers})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


@pytest.mark.parametrize("forwarded", [
    {"X-Forwarded-For": "127.0.0.1"},           # pass-through proxy, spoofed claim
    {"X-Forwarded-For": "203.0.113.9"},         # honest proxy, real remote address
    {"X-Real-IP": "203.0.113.9"},               # proxy that only sets X-Real-IP
])
def test_a_remote_caller_behind_a_local_proxy_is_not_local(live, forwarded) -> None:
    status, body = _post(live, forwarded)
    assert _is_identity_refusal(status, body), (status, body[:200])


def test_a_direct_local_caller_still_writes_uncredentialed(live) -> None:
    status, body = _post(live, {})
    assert not _is_identity_refusal(status, body), (status, body[:200])


@pytest.mark.parametrize("claimed", ["203.0.113.9", "127.0.0.1"])
def test_a_named_proxy_still_cannot_make_a_caller_local(live_behind_named_proxy,
                                                        claimed) -> None:
    status, body = _post(live_behind_named_proxy, {"X-Forwarded-For": claimed})
    assert _is_identity_refusal(status, body), (status, body[:200])
