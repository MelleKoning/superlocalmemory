# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A web page on another domain cannot reach SLM through the browser.

DNS rebinding lets a site point its own domain at 127.0.0.1, so the browser
treats SLM as part of that site and the page could read recall results and
the dashboard token. The browser still names the page's domain in ``Host``;
SLM refuses any request not addressed to this machine.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from superlocalmemory.server.host_guard import HostGuardMiddleware, host_allowed


@pytest.mark.parametrize("host", [
    "localhost", "localhost:8765", "LOCALHOST:8765", "dash.localhost:8765",
    "127.0.0.1:8765", "127.0.0.2", "[::1]:8765", "::1", "[::ffff:127.0.0.1]:8765",
    "192.168.1.10:8765", "10.0.0.5", "",
])
def test_this_machine_and_plain_addresses_are_served(host):
    assert host_allowed(host)


@pytest.mark.parametrize("host", [
    "evil.example:8765", "evil.example", "EVIL.Example.:8765",
    "localhost.evil.example:8765", "127.0.0.1.evil.example", "mybox.local:8765",
])
def test_any_other_name_is_refused(host, monkeypatch):
    monkeypatch.delenv("SLM_ALLOWED_HOSTS", raising=False)
    assert not host_allowed(host)


def test_a_name_the_owner_configured_is_served(monkeypatch):
    monkeypatch.setenv("SLM_ALLOWED_HOSTS", "mybox.local, Other.Host.")
    assert host_allowed("mybox.local:8765")
    assert host_allowed("other.host")
    assert not host_allowed("evil.example")


def _app():
    async def secret(request):
        return PlainTextResponse("memories")
    return HostGuardMiddleware(Starlette(routes=[Route("/recall", secret)]))


def test_a_rebinding_page_gets_nothing():
    client = TestClient(_app())
    response = client.get("/recall", headers={"Host": "evil.example:8765"})
    assert response.status_code == 421
    assert "memories" not in response.text


def test_the_dashboard_on_this_machine_is_unaffected():
    client = TestClient(_app())
    assert client.get("/recall", headers={"Host": "127.0.0.1:8765"}).text == "memories"
    assert client.get("/recall", headers={"Host": "localhost:8765"}).text == "memories"


def test_both_services_put_the_guard_in_front():
    import inspect

    from superlocalmemory.server import ui, unified_daemon

    for module in (unified_daemon, ui):
        assert "HostGuardMiddleware" in inspect.getsource(module), module.__name__


def test_the_real_service_refuses_a_rebinding_host_and_serves_this_machine(
        tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SLM_ALLOWED_HOSTS", raising=False)
    from superlocalmemory.server.unified_daemon import create_app

    client = TestClient(create_app())  # no lifespan: the guard runs before any route
    refused = client.get("/internal/token", headers={"Host": "evil.example:8765"})
    assert refused.status_code == 421
    assert "token" not in refused.json()
    served = client.get("/internal/token", headers={"Host": "127.0.0.1:8765"})
    assert served.status_code != 421


def test_a_name_already_allowed_for_lan_mcp_access_keeps_working(monkeypatch):
    """LAN setups documented before this guard list host names, with a port
    wildcard, in SLM_MCP_ALLOWED_HOSTS. They must keep working unchanged."""
    monkeypatch.delenv("SLM_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.144:*,slm.lan:*,*.office.lan")
    assert host_allowed("slm.lan:8765")
    assert host_allowed("box.office.lan:8765")
    assert not host_allowed("office.lan.evil.example")
    assert not host_allowed("evil.example")


def test_a_wildcard_the_owner_chose_allows_every_name(monkeypatch):
    monkeypatch.delenv("SLM_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "*")
    assert host_allowed("anything.example:8765")
