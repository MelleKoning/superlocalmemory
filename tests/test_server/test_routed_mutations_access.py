# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Who may replace and review in a routed profile, and the MCP tools that reach it.

Company mode, real canonical writer: the caller's role on the ROUTED profile is
what counts, for the permission check and for the correction policy alike. A
role held on the active profile must never stand in for it.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.test_server.test_per_request_profile import _core_tools

OLD = "The release train for the platform team leaves on Tuesday at 14:00 UTC."
NEW = "The release train for the platform team leaves on Thursday at 09:00 UTC."


@contextmanager
def _company_daemon(engine):
    """Real writer + RBAC store, company mode forced for the policy layer."""
    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
        M042_correction_case_ledger,
    )

    engine.profile_id = "default"
    engine._config.active_profile = "default"
    with engine._db.raw_connection() as conn:
        for migration in (M018_ingestion_operations, M032_write_coordinator_admission,
                          M033_projection_transactions, M034_obligation_integrity,
                          M042_correction_case_ledger):
            migration.apply(conn)
        for profile_id in ("default", "work"):
            conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                         (profile_id, profile_id))
        conn.commit()
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    app.state.rbac = RbacEngine(str(engine._config.db_path))
    app.state.deployment = SimpleNamespace(is_enterprise=True)
    bind_profile_runtime(app.state, engine, engine._config)
    runtime = CanonicalRememberRuntime.for_engine(engine)
    runtime.start()
    app.state.canonical_remember_runtime = runtime
    descriptor = app.state.daemon_descriptor
    headers = {"X-SLM-Daemon-Capability": descriptor.capability,
               "X-SLM-Target-Instance": descriptor.instance_id}
    try:
        yield TestClient(app), headers, app
    finally:
        runtime.stop()


@pytest.fixture
def company(engine_with_mock_deps):
    with _company_daemon(engine_with_mock_deps) as triple:
        yield triple


def _user(client, headers, app, name: str, roles: dict[str, str]) -> dict[str, str]:
    """A logged-in user holding ``roles`` (profile -> role); returns its headers."""
    client.post("/api/rbac/users", json={"username": name, "password": "password-1234",
                                         "role": roles.get("default", "viewer")},
                headers=headers)
    rbac = app.state.rbac
    user_id = {u["username"]: u["user_id"] for u in rbac.list_users()}[name]
    for profile_id, role in roles.items():
        rbac.set_membership(profile_id, user_id, role)
    return {**headers, "X-SLM-User-Session": rbac.create_session(user_id)}


def _owner_replace(client, headers, tag: str) -> tuple[str, dict]:
    """The machine owner saves and replaces in 'work'; returns (old_id, case)."""
    old = client.post("/remember", json={"content": OLD, "profile_id": "work",
                                         "idempotency_key": f"{tag}-old"}, headers=headers)
    assert old.status_code == 200, old.text
    [old_id] = old.json()["fact_ids"]
    new = client.post("/remember", json={"content": NEW, "profile_id": "work",
                                         "replaces": old_id,
                                         "idempotency_key": f"{tag}-new"}, headers=headers)
    assert new.status_code == 200, new.text
    assert new.json()["replaced"]["ok"] is True, new.json()["replaced"]
    [case] = new.json()["replaced"]["cases"]
    return old_id, case


def _rollback(client, headers, case: dict, profile: str = "work"):
    return client.post(f"/api/corrections/{case['case_id']}/rollback",
                       json={"expected_version": case["version"], "profile_id": profile},
                       headers=headers)


def _status(app, case: dict) -> str:
    rows = app.state.engine._db.execute(
        "SELECT status FROM correction_cases WHERE case_id=?", (case["case_id"],))
    return dict(rows[0])["status"]


def test_an_admin_of_the_routed_profile_can_roll_back_there(company) -> None:
    """Positive control: without it, every 403 below could be a broken harness."""
    client, headers, app = company
    _old, case = _owner_replace(client, headers, "acl-admin")
    admin = _user(client, headers, app, "work-admin", {"work": "admin"})
    response = _rollback(client, admin, case)
    assert response.status_code == 200, response.text
    assert _status(app, case) == "rolled_back"


def test_an_admin_elsewhere_cannot_roll_back_where_they_are_only_a_member(company) -> None:
    """The correction policy must read the role on 'work', not on 'default'."""
    client, headers, app = company
    _old, case = _owner_replace(client, headers, "acl-escalate")
    member = _user(client, headers, app, "mixed",
                   {"default": "admin", "work": "member"})
    response = _rollback(client, member, case)
    assert response.status_code == 403, response.text
    assert _status(app, case) == "applied"


def test_a_viewer_cannot_list_another_profiles_corrections(company) -> None:
    client, headers, app = company
    _owner_replace(client, headers, "acl-list")
    viewer = _user(client, headers, app, "work-viewer",
                   {"default": "admin", "work": "viewer"})
    response = client.get("/api/corrections", params={"profile_id": "work"},
                          headers=viewer)
    assert response.status_code == 403, response.text


def test_a_member_of_the_routed_profile_cannot_replace_there(company) -> None:
    """Saving is open to members, retiring is not -- judged on 'work'."""
    client, headers, app = company
    old = client.post("/remember", json={"content": OLD, "profile_id": "work",
                                         "idempotency_key": "acl-member-old"},
                      headers=headers)
    [old_id] = old.json()["fact_ids"]
    member = _user(client, headers, app, "work-member",
                   {"default": "admin", "work": "member"})
    engine = app.state.engine
    before = len(engine._db.execute("SELECT 1 FROM memories"))
    refused = client.post("/remember", json={"content": NEW, "profile_id": "work",
                                             "replaces": old_id,
                                             "idempotency_key": "acl-member-new"},
                          headers=member)
    assert refused.status_code == 403, refused.text
    assert len(engine._db.execute("SELECT 1 FROM memories")) == before


def test_permission_comes_before_existence_for_routed_review(company) -> None:
    """A caller without access learns nothing about which profiles exist."""
    client, headers, app = company
    outsider = _user(client, headers, app, "outsider", {"default": "admin"})
    body = {"expected_version": 1, "profile_id": "ghost"}
    denied = client.post("/api/corrections/abc/rollback", json=body, headers=outsider)
    owner = client.post("/api/corrections/abc/rollback", json=body, headers=headers)
    assert denied.status_code == 403, denied.text
    assert owner.status_code == 404, owner.text


# ---------------------------------------------------------------------------
# MCP: profile_id reaches the daemon only when the caller set it
# ---------------------------------------------------------------------------

def _capture(monkeypatch, reply: dict) -> dict:
    import superlocalmemory.cli.daemon as daemon

    captured: dict = {}

    def request(method, path, body=None, **kwargs):
        captured.update(method=method, path=path, body=body)
        return reply

    monkeypatch.setattr(daemon, "is_daemon_running", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "daemon_request", request)
    return captured


@pytest.mark.parametrize("profile_id,expected", [("work", "work"), ("  work ", "work")])
def test_review_correction_tool_routes_profile_id(monkeypatch, profile_id, expected) -> None:
    import asyncio

    captured = _capture(monkeypatch, {"success": True, "correction_case": {}})
    tool = _core_tools()["review_correction"]
    result = asyncio.run(tool("abc", "rollback", 1, profile_id=profile_id))
    assert result["success"] is True, result
    assert captured["body"]["profile_id"] == expected


def test_review_correction_tool_legacy_call_is_unchanged(monkeypatch) -> None:
    import asyncio

    captured = _capture(monkeypatch, {"success": True, "correction_case": {}})
    tool = _core_tools()["review_correction"]
    for profile_id in ("", "   "):
        asyncio.run(tool("abc", "rollback", 1, profile_id=profile_id))
        assert captured["body"] == {"expected_version": 1}
    asyncio.run(tool("abc", "rollback", 1))
    assert captured["body"] == {"expected_version": 1}


def test_list_corrections_tool_routes_profile_id_only_when_set(monkeypatch) -> None:
    import asyncio

    captured = _capture(monkeypatch, {"success": True, "corrections": []})
    tool = _core_tools()["list_corrections"]
    asyncio.run(tool(limit=5, profile_id="work"))
    assert captured["path"] == "/api/corrections?limit=5&profile_id=work"
    asyncio.run(tool(limit=5))
    assert captured["path"] == "/api/corrections?limit=5"


def test_correction_tools_expose_profile_id_as_optional() -> None:
    from unittest.mock import MagicMock

    from superlocalmemory.mcp.http_transport import SLMFastMCP
    from superlocalmemory.mcp.tools_core import register_core_tools

    srv = SLMFastMCP("schema probe")
    register_core_tools(srv, MagicMock())
    tools = {t.name: t for t in srv._tool_manager.list_tools()}
    for name in ("review_correction", "list_corrections"):
        schema = tools[name].parameters
        assert "profile_id" in schema["properties"], name
        assert "profile_id" not in schema.get("required", []), name
