# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The dashboard summary and the timeline need READ on the workspace.

The attack (found against a running daemon in company mode): with team
accounts on and login required, ``GET /api/summary`` and
``GET /api/v3/timeline/`` returned memory text to a caller with no session,
to a signed-in user with no role on the workspace, and to a member of another
workspace. ``GET /api/memories`` refused all three. Both paths were missing
from the list of reads that need READ.
"""

from __future__ import annotations

import pytest

from tests.test_server.test_facets_rbac import _client

_SECRET = "Zeus"
_PATHS = ("/api/summary", "/api/v3/timeline/", "/api/v3/timeline")


@pytest.fixture()
def company(tmp_path):
    client, _app, _db, rbac = _client(tmp_path)
    owner = rbac.create_user("alice", "pw-123456789", "Alice")
    rbac.set_membership("default", owner["user_id"], "admin")
    rbac.set_require_login(True)
    return client, rbac


def _get(client, path: str, session: str | None = None):
    headers = {"Host": "127.0.0.1:8765"}
    if session:
        headers["X-SLM-User-Session"] = session
    return client.get(path, headers=headers)


@pytest.mark.parametrize("path", _PATHS)
def test_no_session_in_company_mode_is_refused(company, path) -> None:
    client, _rbac = company
    resp = _get(client, path)
    assert resp.status_code == 401, resp.text[:200]
    assert _SECRET not in resp.text


@pytest.mark.parametrize("path", _PATHS)
def test_a_member_of_another_workspace_is_refused(company, path) -> None:
    client, rbac = company
    other = rbac.create_user("olga", "pw-123456789", "Olga")
    rbac.set_membership("clientx", other["user_id"], "member")
    resp = _get(client, path, rbac.create_session(other["user_id"]))
    assert resp.status_code == 403, resp.text[:200]
    assert _SECRET not in resp.text


@pytest.mark.parametrize("path", _PATHS)
def test_summary_and_timeline_are_classed_as_reads(path) -> None:
    from superlocalmemory.server import read_gates

    assert read_gates.is_sensitive_dashboard_read("GET", path)


# ``GET /list`` (the agent door behind ``slm list``) returned the newest
# memories to the same three callers while ``GET /recall`` refused them.


def test_list_without_a_session_in_company_mode_is_refused(company) -> None:
    client, _rbac = company
    resp = _get(client, "/list")
    assert resp.status_code == 401, resp.text[:200]
    assert _SECRET not in resp.text


def test_list_for_a_member_of_another_workspace_is_refused(company) -> None:
    client, rbac = company
    other = rbac.create_user("olga", "pw-123456789", "Olga")
    rbac.set_membership("clientx", other["user_id"], "member")
    resp = _get(client, "/list", rbac.create_session(other["user_id"]))
    assert resp.status_code == 403, resp.text[:200]
    assert _SECRET not in resp.text


def test_list_for_a_reader_of_the_workspace_still_answers(company) -> None:
    client, rbac = company
    viewer = rbac.create_user("vera", "pw-123456789", "Vera")
    rbac.set_membership("default", viewer["user_id"], "viewer")
    resp = _get(client, "/list", rbac.create_session(viewer["user_id"]))
    assert resp.status_code == 200, resp.text[:200]
    assert _SECRET in resp.text


def test_list_in_personal_mode_still_answers(tmp_path) -> None:
    client, _app, _db, _rbac = _client(tmp_path)
    resp = _get(client, "/list")
    assert resp.status_code == 200, resp.text[:200]
    assert _SECRET in resp.text
