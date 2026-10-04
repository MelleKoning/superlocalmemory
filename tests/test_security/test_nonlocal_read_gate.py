# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A computer on the LAN cannot read memory without credentials.

The attack: the daemon is bound to a LAN address (``SLM_DAEMON_HOST=0.0.0.0``).
Before 4.1.20 any machine that could reach the port read memories with a plain
``GET /api/memories`` and no credentials: the read gate was a no-op without
team accounts and the API-key check let every read through.

The route-table test walks every route the daemon serves and sends the
request an uncredentialed LAN caller would send. Anything that is not refused
must be on the reviewed lists in ``server/access_gate.py``, so a route added
later cannot be silently open.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from superlocalmemory.server import access_gate
from superlocalmemory.server.unified_daemon import create_app

_LAN_PEER = ("192.168.50.20", 50000)
_BASE = "http://192.168.50.144:8765"

#: The reviewed exceptions, by route path. Changing this set is a security
#: review: every entry must serve no memory content or check credentials itself.
_REVIEWED_OPEN = {
    "/": "dashboard shell (static HTML, no data)",
    "/health": "liveness probe",
    "/favicon.ico": "redirect to a static icon",
    "/static": "packaged HTML/JS/CSS",
    "/mcp": "own check: network callers need HTTPS and a remote key (remote_access)",
}


@pytest.fixture(scope="module")
def app():
    return create_app()


def _lan(app, **kw) -> TestClient:
    return TestClient(app, client=kw.pop("client", _LAN_PEER), base_url=_BASE,
                      raise_server_exceptions=False, **kw)


def _refused(resp) -> bool:
    return resp.status_code == 401 and resp.json().get("error") == "remote_auth_required"


def test_lan_peer_cannot_read_memories_without_credentials(app) -> None:
    resp = _lan(app).get("/api/memories")
    assert _refused(resp), (resp.status_code, resp.text[:200])
    message = resp.json()["message"]
    for hint in ("X-SLM-API-Key", "SLM_REMOTE=1", "SLM_MCP_ALLOWED_HOSTS", "sign in"):
        assert hint in message


def test_a_proxied_loopback_read_is_a_network_read(app) -> None:
    client = _lan(app, client=("127.0.0.1", 50000))
    resp = client.get("/api/memories", headers={"X-Forwarded-For": "192.168.50.20"})
    assert _refused(resp), resp.text[:200]


def test_a_local_read_is_unchanged(app) -> None:
    resp = _lan(app, client=("127.0.0.1", 50000)).get("/api/memories")
    assert not _refused(resp)


def test_the_dashboard_websocket_refuses_a_lan_peer(app) -> None:
    with pytest.raises(WebSocketDisconnect) as closed:
        with _lan(app).websocket_connect("/ws/updates") as ws:
            ws.receive_json()
    assert closed.value.code == 4001


# -- each credential lets a network caller through ---------------------------------


def test_the_api_key_is_a_network_credential(app) -> None:
    from superlocalmemory.infra.auth_middleware import API_KEY_FILE

    key_file = Path(API_KEY_FILE)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text("lan-reader-key-0123456789")
    try:
        client = _lan(app)
        assert not _refused(client.get("/api/memories",
                                       headers={"X-SLM-API-Key": "lan-reader-key-0123456789"}))
        assert _refused(client.get("/api/memories", headers={"X-SLM-API-Key": "wrong"}))
    finally:
        key_file.unlink()


def test_the_install_token_is_not_a_network_credential(app) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    token = ensure_install_token()
    resp = _lan(app).get("/api/memories", headers={"X-Install-Token": token})
    assert _refused(resp)


def test_an_allowlisted_lan_dashboard_still_reads(app, monkeypatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.0/24")
    assert not _refused(_lan(app).get("/api/memories"))
    other = _lan(app, client=("10.9.9.9", 50000))
    assert _refused(other.get("/api/memories"))


def test_a_team_session_is_a_network_credential(app, monkeypatch) -> None:
    sessions = {"good-session": {"user_id": "u1"}}
    rbac = SimpleNamespace(resolve_session=lambda tok: sessions.get(tok),
                           user_count=lambda: 1, require_login=lambda: False,
                           has_permission=lambda *a: True)
    monkeypatch.setattr(app.state, "rbac", rbac, raising=False)
    client = _lan(app)
    assert not _refused(client.get("/api/memories",
                                   headers={"X-SLM-User-Session": "good-session"}))
    assert not _refused(client.get("/api/memories",
                                   cookies={"slm_session": "good-session"}))
    assert _refused(client.get("/api/memories", headers={"X-SLM-User-Session": "forged"}))


def test_the_mesh_secret_opens_mesh_routes_only(app, monkeypatch) -> None:
    broker = SimpleNamespace(_shared_secret="fleet-secret-0123456789")
    monkeypatch.setattr(app.state, "mesh_broker", broker, raising=False)
    client = _lan(app)
    headers = {"X-Mesh-Secret": "fleet-secret-0123456789"}
    assert not _refused(client.get("/mesh/status", headers=headers))
    assert _refused(client.get("/api/memories", headers=headers))
    assert _refused(client.get("/mesh/status", headers={"X-Mesh-Secret": "wrong"}))


# -- the route table: nothing reachable that is not reviewed -------------------------


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}/]+\}", "x", path).replace(":path}", "}")


def _walk(routes, prefix: str = ""):
    """Flatten the route tree, through FastAPI's included-router wrappers."""
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:  # FastAPI >= 0.13x wraps include_router()
            yield from _walk(included.routes, prefix + route.include_context.prefix)
        else:
            yield prefix, route


def _routes(app) -> list[tuple[str, str, str]]:
    """(kind, method, concrete path) for every route the daemon serves."""
    out: list[tuple[str, str, str]] = []
    for prefix, route in _walk(app.routes):
        path = _concrete(prefix + getattr(route, "path", ""))
        if isinstance(route, WebSocketRoute):
            out.append(("ws", "WS", path))
        elif isinstance(route, Mount):
            out.append(("http", "GET", path + "/probe"))
        elif isinstance(route, Route):
            for method in sorted(route.methods or {"GET"}):
                if method != "HEAD":
                    out.append(("http", method, path))
    return out


def _reviewed(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _REVIEWED_OPEN)


def _lan_scope(kind: str, path: str) -> dict:
    return {"type": "websocket" if kind == "ws" else "http", "path": path,
            "client": _LAN_PEER, "headers": [(b"host", b"192.168.50.144:8765")]}


def test_every_route_refuses_an_uncredentialed_lan_caller_unless_reviewed(app) -> None:
    """Asks the gate about every route instead of calling the routes.

    Calling them would be the stronger test only while the gate works: if it
    ever regressed, the walk would run every handler in the daemon, including
    stop and restart. The wiring itself is proven by the request-level tests
    above and by ``test_the_gate_wraps_every_route``.
    """
    routes = _routes(app)
    assert len(routes) > 150, "the route walk found too few routes to be the daemon"
    open_routes = sorted({
        f"{method} {path}" for kind, method, path in routes
        if access_gate.network_caller_allowed(_lan_scope(kind, path), app.state)
        and not _reviewed(path)
    })
    assert open_routes == [], (
        "reachable from the LAN without credentials and not reviewed: "
        + ", ".join(open_routes))


def test_the_gate_wraps_every_route(app) -> None:
    """Installed in the real app, outside the auth, rate-limit and route code,
    and inside only the host and forwarded-header guards."""
    from superlocalmemory.server.forwarded_guard import ForwardedLoopbackDemotion
    from superlocalmemory.server.host_guard import HostGuardMiddleware

    order = [m.cls for m in app.user_middleware]  # outermost first
    assert order[:3] == [HostGuardMiddleware, ForwardedLoopbackDemotion,
                         access_gate.NonLocalAccessGate]


def test_the_mcp_mount_answers_for_itself(app) -> None:
    client = _lan(app)
    # 4.1.20: plain HTTP from another computer is refused before the key
    # check (remote_requires_tls); over HTTPS a missing key is 401.
    assert client.post("/mcp/", json={}).status_code == 403
    assert client.post("/mcp/hermes", json={}).status_code == 403
    secure = TestClient(app, client=_LAN_PEER, base_url="https://192.168.50.144:8765",
                        raise_server_exceptions=False)
    assert secure.post("/mcp/", json={}).status_code == 401
    assert secure.post("/mcp/hermes", json={}).status_code == 401


def test_the_gate_and_the_reviewed_list_agree() -> None:
    """The code's open paths are exactly what this test reviewed."""
    for path in access_gate.PUBLIC_PATHS:
        assert _reviewed(path)
    for prefix in access_gate.PUBLIC_PREFIXES + access_gate.SELF_GATED_PREFIXES:
        if prefix.startswith(("/v1/", "/v1beta/")):
            continue  # LLM proxy: not mounted unless enabled; see access_gate
        assert _reviewed(prefix.rstrip("/"))
