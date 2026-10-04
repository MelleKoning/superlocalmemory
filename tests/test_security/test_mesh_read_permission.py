# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Mesh reads respect workspace permissions (deferred from 4.1.19).

The attacks, before 4.1.20:

* with team accounts and login required, ``GET /mesh/*`` checked only that the
  caller was a machine allowed to use the mesh, so a signed-in user without
  READ, or the dashboard with no session at all, read peers, inboxes, shared
  state and locks;
* ``GET /mesh/state/delta?profile=<any>`` and ``/mesh/lock/delta?profile=``
  let a fleet node (anyone holding the fleet-wide mesh secret) read any
  workspace on the machine, not just the one this node serves.
"""

from __future__ import annotations

import sqlite3

import pytest
from starlette.testclient import TestClient

from tests.test_server.test_facets_rbac import _client

_SECRET_STATE = "deploy-window-is-friday"
_FLEET = "fleet-secret-0123456789abcdef"
_HOST = {"Host": "127.0.0.1:8765"}


def _broker(tmp_path):
    from superlocalmemory.mesh.broker import MeshBroker
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL,
        _MESH_V346_ALTERS,
        _MESH_V346_DDL,
    )

    db_path = tmp_path / "mesh.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_MESH_DDL)
    for statement in _MESH_V346_ALTERS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            pass
    conn.executescript(_MESH_V346_DDL)
    conn.commit()
    conn.close()
    broker = MeshBroker(db_path)
    broker.set_state("plan", _SECRET_STATE, "agent-a", profile_id="default")
    broker.set_state("plan", "work-" + _SECRET_STATE, "agent-b", profile_id="work")
    return broker


@pytest.fixture()
def mesh(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_MESH_SHARED_SECRET", _FLEET)
    client, app, _db, rbac = _client(tmp_path)
    app.state.mesh_broker = _broker(tmp_path)
    return client, app, rbac


def _company(rbac):
    admin = rbac.create_user("alice", "pw-123456789", "Alice")
    rbac.set_membership("default", admin["user_id"], "admin")
    rbac.set_require_login(True)


def _install_token() -> dict[str, str]:
    from superlocalmemory.core.security_primitives import ensure_install_token

    return {"X-Install-Token": ensure_install_token()}


def test_company_mode_without_a_session_cannot_read_mesh_state(mesh) -> None:
    client, _app, rbac = mesh
    _company(rbac)
    resp = client.get("/mesh/state", headers={**_HOST, **_install_token()})
    assert resp.status_code == 401, resp.text[:200]
    assert _SECRET_STATE not in resp.text


@pytest.mark.parametrize("path", ["/mesh/state", "/mesh/peers", "/mesh/events",
                                  "/mesh/status", "/mesh/state/delta", "/mesh/lock/delta",
                                  "/mesh/inbox/p1", "/mesh/pending/p1", "/mesh/state/plan"])
def test_a_role_without_read_cannot_read_the_mesh(mesh, path) -> None:
    client, _app, rbac = mesh
    _company(rbac)
    nobody = rbac.create_user("nemo", "pw-123456789", "Nemo")
    resp = client.get(path, headers={**_HOST, **_install_token(),
                                     "X-SLM-User-Session": rbac.create_session(nobody["user_id"])})
    assert resp.status_code == 403, resp.text[:200]
    assert _SECRET_STATE not in resp.text


def test_a_reader_reads_only_the_workspaces_it_may_read(mesh) -> None:
    client, _app, rbac = mesh
    _company(rbac)
    viewer = rbac.create_user("vera", "pw-123456789", "Vera")
    rbac.set_membership("default", viewer["user_id"], "viewer")
    headers = {**_HOST, **_install_token(),
               "X-SLM-User-Session": rbac.create_session(viewer["user_id"])}
    own = client.get("/mesh/state", headers=headers)
    assert own.status_code == 200 and _SECRET_STATE in own.text
    assert client.get("/mesh/lock/delta", headers=headers).status_code == 200
    for path in ("/mesh/state/delta", "/mesh/lock/delta"):
        other = client.get(path, params={"profile": "work"}, headers=headers)
        assert other.status_code == 403, (path, other.text[:200])
        assert "work-" + _SECRET_STATE not in other.text


def test_this_computers_own_agents_still_read_in_company_mode(mesh) -> None:
    """The MCP mesh tools call with the daemon capability: a program, not a person."""
    client, app, rbac = mesh
    _company(rbac)
    descriptor = app.state.daemon_descriptor
    local = TestClient(app, client=("127.0.0.1", 40000), base_url="http://127.0.0.1:8765")
    resp = local.get("/mesh/state", headers={
        **_HOST, "X-SLM-Daemon-Capability": descriptor.capability,
        "X-SLM-Target-Instance": str(descriptor.instance_id)})
    assert resp.status_code == 200, resp.text[:200]


def _fleet_node(app) -> TestClient:
    return TestClient(app, client=("192.168.50.31", 40000),
                      base_url="http://192.168.50.144:8765")


@pytest.mark.parametrize("path", ["/mesh/state/delta", "/mesh/lock/delta"])
def test_a_fleet_node_cannot_read_another_workspace(mesh, path) -> None:
    _client_, app, _rbac = mesh
    node = _fleet_node(app)
    resp = node.get(path, params={"profile": "work"}, headers={"X-Mesh-Secret": _FLEET})
    assert resp.status_code == 403, resp.text[:200]
    assert "work-" + _SECRET_STATE not in resp.text


def test_a_fleet_node_still_syncs_the_workspace_this_node_serves(mesh) -> None:
    _client_, app, rbac = mesh
    _company(rbac)  # fleet sync keeps working in company mode
    node = _fleet_node(app)
    resp = node.get("/mesh/state", headers={"X-Mesh-Secret": _FLEET})
    assert resp.status_code == 200, resp.text[:200]
    assert _SECRET_STATE in resp.text
    locks = node.get("/mesh/lock/delta", headers={"X-Mesh-Secret": _FLEET})
    assert locks.status_code == 200, locks.text[:200]


def test_personal_mode_is_unchanged(mesh) -> None:
    client, _app, _rbac = mesh
    resp = client.get("/mesh/state", headers={**_HOST, **_install_token()})
    assert resp.status_code == 200 and _SECRET_STATE in resp.text
