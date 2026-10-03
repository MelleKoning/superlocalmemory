# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L3-01: GET /api/v3/facets must gate on Permission.READ like its siblings.

Before this fix the route had no RBAC check at all and was absent from
``_SENSITIVE_READ_*``, so an anonymous caller in a company-mode workspace
(``require_login`` on) could enumerate every project name and every agent id
that ever saved a memory — while the sibling routes (``/api/memories``,
``/api/memory-kinds/status``, ``/api/upgrade/status``) all refused the same
caller with 401.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from superlocalmemory.access.rbac import RbacEngine
from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M024_rbac_users_roles as m024
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


def _client(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    conn = sqlite3.connect(str(db_path))
    m024.apply(conn)
    conn.commit()
    conn.close()
    db = DatabaseManager(db_path)
    mid = db.store_memory(MemoryRecord(
        profile_id="default", content="x",
        metadata={"project": "secret-acquisition-zeus", "agent_id": "claude-desktop"},
    ))
    db.store_fact(AtomicFact(
        profile_id="default", memory_id=mid, content="Board approved the Zeus deal.",
        fact_type=FactType.SEMANTIC,
    ))
    rbac = RbacEngine(str(db_path))
    app = create_app()
    app.state.engine = SimpleNamespace(_db=db, db=db, profile_id="default",
                                       _profile_id="default")
    app.state.rbac = rbac
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    return client, app, db, rbac


def test_facets_401_for_anonymous_in_company_mode(tmp_path):
    client, app, db, rbac = _client(tmp_path)
    rbac.create_user("alice", "pw-123456789", "Alice")
    rbac.set_require_login(True)
    r = client.get("/api/v3/facets", headers={"Host": "127.0.0.1:8765"})
    assert r.status_code == 401, r.text
    assert "secret-acquisition-zeus" not in r.text


def test_facets_403_for_a_role_without_read(tmp_path):
    client, app, db, rbac = _client(tmp_path)
    user = rbac.create_user("vera", "pw-123456789", "Vera")
    rbac.set_membership("default", user["user_id"], "viewer")
    rbac.set_require_login(True)
    # Give viewer a profile-scoped membership but then verify a role with NO
    # grant at all (no membership row) also gets 403, not a silent 200.
    nobody = rbac.create_user("nemo", "pw-123456789", "Nemo")
    session = rbac.create_session(nobody["user_id"])
    r = client.get("/api/v3/facets", headers={
        "Host": "127.0.0.1:8765", "X-SLM-User-Session": session,
    })
    assert r.status_code == 403, r.text
    assert "secret-acquisition-zeus" not in r.text


def test_facets_200_for_a_member_with_read(tmp_path):
    client, app, db, rbac = _client(tmp_path)
    user = rbac.create_user("vera", "pw-123456789", "Vera")
    rbac.set_membership("default", user["user_id"], "viewer")
    rbac.set_require_login(True)
    session = rbac.create_session(user["user_id"])
    r = client.get("/api/v3/facets", headers={
        "Host": "127.0.0.1:8765", "X-SLM-User-Session": session,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["projects"][0]["name"] == "secret-acquisition-zeus"
