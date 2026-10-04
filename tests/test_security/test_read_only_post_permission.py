# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Search and memory chat need READ on the workspace, like every other read.

The attack: with team accounts on and login required (company mode), the read
permission check ran for GET requests only. ``POST /api/search`` and the
memory chat (``POST /api/v3/chat/stream``) return memory content, so a caller
with no session, or a signed-in user whose role grants no READ on the
workspace, could read memories through them while ``GET /api/memories``
refused the same caller.

``POST /api/v3/recall/trace`` has the same shape; it is fixed with the Answer
Check work and is deliberately not covered here.
"""

from __future__ import annotations

import pytest

from tests.test_server.test_facets_rbac import _client

_SECRET = "Zeus"
_SEARCH = ("/api/search", {"query": "Zeus deal", "limit": 5})
_CHAT = ("/api/v3/chat/stream", {"query": "What was approved?", "mode": "a"})


@pytest.fixture()
def company(tmp_path):
    client, app, db, rbac = _client(tmp_path)
    owner = rbac.create_user("alice", "pw-123456789", "Alice")
    rbac.set_membership("default", owner["user_id"], "admin")
    rbac.set_require_login(True)
    return client, rbac


def _post(client, route, session: str | None = None):
    headers = {"Host": "127.0.0.1:8765"}
    if session:
        headers["X-SLM-User-Session"] = session
    return client.post(route[0], json=route[1], headers=headers)


@pytest.mark.parametrize("route", [_SEARCH, _CHAT], ids=["search", "chat"])
def test_no_session_in_company_mode_is_refused(company, route) -> None:
    client, _rbac = company
    resp = _post(client, route)
    assert resp.status_code == 401, resp.text[:200]
    assert _SECRET not in resp.text


@pytest.mark.parametrize("route", [_SEARCH, _CHAT], ids=["search", "chat"])
def test_a_role_without_read_is_refused(company, route) -> None:
    client, rbac = company
    nobody = rbac.create_user("nemo", "pw-123456789", "Nemo")  # no membership
    resp = _post(client, route, rbac.create_session(nobody["user_id"]))
    assert resp.status_code == 403, resp.text[:200]
    assert _SECRET not in resp.text


def test_a_member_with_read_still_searches(company) -> None:
    client, rbac = company
    viewer = rbac.create_user("vera", "pw-123456789", "Vera")
    rbac.set_membership("default", viewer["user_id"], "viewer")
    resp = _post(client, _SEARCH, rbac.create_session(viewer["user_id"]))
    assert resp.status_code not in (401, 403), resp.text[:200]


def test_personal_mode_owner_still_searches(tmp_path) -> None:
    client, app, db, rbac = _client(tmp_path)
    resp = _post(client, _SEARCH)
    assert resp.status_code not in (401, 403), resp.text[:200]


def test_search_and_chat_are_classed_as_reads() -> None:
    from superlocalmemory.server import read_gates

    assert read_gates.is_sensitive_dashboard_read("POST", "/api/search")
    assert read_gates.is_sensitive_dashboard_read("POST", "/api/v3/chat/stream")
