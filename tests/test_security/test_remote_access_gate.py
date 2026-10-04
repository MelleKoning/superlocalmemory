# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""MCP from another computer: HTTPS, a named key, never company mode.

Every test here sends the request a hostile or careless caller on the network
would send and asserts the daemon's answer, through the real application.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from superlocalmemory.server.remote_keys import RemoteKeyStore
from superlocalmemory.server.remote_listener import RemoteListenerASGI
from superlocalmemory.server.unified_daemon import create_app

_LAN_PEER = ("192.168.50.20", 50000)
_ACCEPT = {"Accept": "application/json, text/event-stream"}


def _call(tool: str, arguments: dict | None = None, rid: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments or {}}}


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture()
def keys():
    store = RemoteKeyStore()
    _, write_secret = store.add("hermes-laptop", "write")
    _, read_secret = store.add("viewer", "read")
    return SimpleNamespace(store=store, write=write_secret, read=read_secret)


def _main(app, scheme: str = "https", client=_LAN_PEER) -> TestClient:
    """The main listener, as a network caller sees it."""
    return TestClient(app, client=client, base_url=f"{scheme}://192.168.50.144:8765",
                      raise_server_exceptions=False)


def _remote(app, scheme: str = "https", client=("127.0.0.1", 50000)) -> TestClient:
    """The remote listener (TLS is simulated by the scheme)."""
    return TestClient(RemoteListenerASGI(app, ("localhost", "127.0.0.1")), client=client,
                      base_url=f"{scheme}://127.0.0.1:8443", raise_server_exceptions=False)


def _bearer(secret: str) -> dict:
    return {**_ACCEPT, "Authorization": f"Bearer {secret}"}


def _gate_passed(resp) -> bool:
    """The request got past the remote gate (the MCP app itself answered)."""
    if resp.status_code in (401, 403):
        return False
    try:
        return resp.json().get("error") not in {
            "remote_requires_tls", "remote_auth_required", "remote_mcp_company_mode"}
    except ValueError:
        return True


# -- HTTPS ------------------------------------------------------------------------------


def test_plain_http_remote_mcp_is_refused_by_default(app, keys) -> None:
    resp = _main(app, "http").post("/mcp/hermes", json=_call("recall"), headers=_bearer(keys.write))
    assert resp.status_code == 403
    assert resp.json()["error"] == "remote_requires_tls"
    assert "slm remote keys revoke" in resp.json()["message"]


def test_plaintext_opt_in_allows_the_main_listener_only(app, keys, monkeypatch) -> None:
    monkeypatch.setenv("SLM_REMOTE_ALLOW_PLAINTEXT", "1")
    resp = _main(app, "http").post("/mcp/hermes", json=_call("recall"), headers=_bearer(keys.write))
    assert _gate_passed(resp), resp.text[:300]
    # The remote listener never accepts plain HTTP, opt-in or not.
    resp = _remote(app, "http").post("/mcp/hermes", json=_call("recall"),
                                     headers=_bearer(keys.write))
    assert resp.status_code == 403 and resp.json()["error"] == "remote_requires_tls"


def test_crit_key_replay_over_a_downgraded_connection_is_refused(app, keys) -> None:
    """CRIT 1: a captured valid key replayed over plain HTTP gets nothing."""
    for client in (_main(app, "http"), _remote(app, "http")):
        resp = client.post("/mcp/hermes", json=_call("recall", {"query": "x"}),
                           headers=_bearer(keys.write))
        assert resp.status_code == 403, resp.text[:200]
        assert resp.json()["error"] == "remote_requires_tls"
        assert "result" not in resp.json()


# -- keys -------------------------------------------------------------------------------


def test_no_key_is_401(app) -> None:
    resp = _main(app).post("/mcp/hermes", json=_call("recall"), headers=_ACCEPT)
    assert resp.status_code == 401
    assert resp.json()["error"] == "remote_auth_required"


def test_wrong_bearer_is_401_and_no_secret_reaches_the_logs(app, keys, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    wrong = "slmr_" + "A" * 43
    resp = _main(app).post("/mcp/hermes", json=_call("recall"), headers=_bearer(wrong))
    assert resp.status_code == 401
    resp = _main(app).post("/mcp/hermes", json=_call("recall"), headers=_bearer(keys.write))
    assert _gate_passed(resp)
    logged = caplog.text
    assert wrong not in logged and keys.write not in logged
    for record in keys.store.list():
        assert record.digest not in logged


def test_revoked_key_is_refused_on_the_next_request_without_restart(app, keys) -> None:
    client = _main(app)
    assert _gate_passed(client.post("/mcp/hermes", json=_call("recall"),
                                    headers=_bearer(keys.write)))
    keys.store.revoke("hermes-laptop")
    resp = client.post("/mcp/hermes", json=_call("recall"), headers=_bearer(keys.write))
    assert resp.status_code == 401


@pytest.mark.parametrize("header", ["X-Install-Token", "X-SLM-Hook-Token",
                                    "X-SLM-Daemon-Capability"])
def test_local_credentials_are_never_remote_credentials(app, header) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    token = ensure_install_token()
    resp = _main(app).post("/mcp/hermes", json=_call("recall"),
                           headers={**_ACCEPT, header: token})
    assert resp.status_code == 401, (header, resp.status_code)
    resp = _remote(app).post("/mcp/hermes", json=_call("recall"),
                             headers={**_ACCEPT, header: token})
    assert resp.status_code == 401, (header, resp.status_code)


def test_company_mode_refuses_remote_mcp(app, keys, monkeypatch) -> None:
    import superlocalmemory.core.admission as admission

    monkeypatch.setattr(admission, "_resolve_deployment",
                        lambda: SimpleNamespace(is_enterprise=True))
    resp = _main(app).post("/mcp/hermes", json=_call("recall"), headers=_bearer(keys.write))
    assert resp.status_code == 403
    assert resp.json()["error"] == "remote_mcp_company_mode"


def test_an_unreadable_deployment_mode_fails_closed(app, keys, monkeypatch) -> None:
    import superlocalmemory.core.admission as admission

    def _boom():
        raise RuntimeError("config.toml unreadable")

    monkeypatch.setattr(admission, "_resolve_deployment", _boom)
    resp = _main(app).post("/mcp/hermes", json=_call("recall"), headers=_bearer(keys.write))
    assert resp.status_code == 403


def test_legacy_api_key_is_write_scope_and_policy_bound(app) -> None:
    from pathlib import Path

    from superlocalmemory.infra.auth_middleware import API_KEY_FILE

    Path(API_KEY_FILE).write_text("legacy-secret-value\n")
    headers = {**_ACCEPT, "X-SLM-API-Key": "legacy-secret-value"}
    resp = _main(app).post("/mcp/hermes", json=_call("switch_profile", {"profile": "x"}),
                           headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["isError"] is True
    assert "not available over remote access" in body["result"]["content"][0]["text"]


# -- the remote listener ------------------------------------------------------------------


def test_remote_listener_treats_a_loopback_peer_as_remote(app) -> None:
    resp = _remote(app).post("/mcp/hermes", json=_call("recall"), headers=_ACCEPT)
    assert resp.status_code == 401


@pytest.mark.parametrize("path", ["/api/memories", "/internal/token", "/internal/prewarm",
                                  "/", "/static/index.html", "/ws/updates", "/recall",
                                  "/api/v3/health", "/mesh/peers", "/v1/messages"])
def test_remote_listener_serves_only_mcp_and_health(app, keys, path) -> None:
    client = _remote(app)
    for method in ("get", "post"):
        resp = getattr(client, method)(path, headers=_bearer(keys.write))
        assert resp.status_code == 404, (method, path, resp.status_code)


def test_remote_listener_health_is_the_public_payload(app) -> None:
    resp = _remote(app).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "pid" not in body and "readiness" not in body
    assert body["status"] == "ok"


def test_no_redirect_on_mcp_paths(app, keys) -> None:
    for client in (_remote(app), _main(app)):
        for path in ("/mcp", "/mcp/", "/mcp/hermes"):
            resp = client.post(path, json=_call("recall"), headers=_bearer(keys.write),
                               follow_redirects=False)
            assert not 300 <= resp.status_code < 400, (path, resp.status_code)


def test_remote_listener_hook_token_is_not_accepted_for_prewarm(app) -> None:
    """The hook token never works on the remote listener (it 404s first)."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    resp = _remote(app).post("/internal/prewarm",
                             headers={"X-SLM-Hook-Token": ensure_install_token()})
    assert resp.status_code == 404


def test_principal_reaches_the_mcp_mount(app, keys) -> None:
    """The gate's principal travels in the ASGI scope to the tool policy: a read
    key is refused a write tool by the policy, not by the gate."""
    resp = _main(app).post("/mcp/hermes", json=_call("remember", {"content": "x"}),
                           headers=_bearer(keys.read))
    assert resp.status_code == 200, resp.text[:300]
    text = resp.json()["result"]["content"][0]["text"]
    assert "read-only" in text


def test_local_mcp_is_unchanged(app) -> None:
    """A caller on this computer needs no key (local-first contract)."""
    resp = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8765",
                      raise_server_exceptions=False).post(
        "/mcp/hermes", json=_call("recall"), headers=_ACCEPT)
    assert resp.status_code not in (401, 403)
    try:
        assert "remote" not in json.dumps(resp.json())
    except ValueError:
        pass
