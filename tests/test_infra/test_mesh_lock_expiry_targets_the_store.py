# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Boot clears expired mesh leases in the store the daemon actually uses.

It cleared them in ``canonical_data_root()/memory.db`` -- the default location
-- even when the store was configured elsewhere, so the configured store kept
its dead leases, and the default location got an empty database created in it
(opening a missing SQLite file creates it).
"""

from __future__ import annotations

import ast
import inspect
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from superlocalmemory.infra.self_heal import expire_stale_mesh_locks


def _store_with_expired_lease(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "CREATE TABLE mesh_locks (file_path TEXT PRIMARY KEY, locked_by TEXT,"
            " locked_at TEXT, expires_at TEXT)")
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn.execute("INSERT INTO mesh_locks VALUES ('a.py', 'dead-node', ?, ?)",
                     (past, past))
        conn.commit()
    finally:
        conn.close()


def test_a_missing_store_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "memory.db"
    missing.parent.mkdir()
    assert expire_stale_mesh_locks(missing) == 0
    assert not missing.exists(), "expiring leases created an empty database"


def test_the_configured_store_is_the_one_cleared(tmp_path: Path) -> None:
    configured = tmp_path / "custom" / "memory.db"
    configured.parent.mkdir()
    _store_with_expired_lease(configured)
    assert expire_stale_mesh_locks(configured) == 1


def _boot_mesh_expiry_target() -> str:
    """The expression the daemon's boot passes to expire_stale_mesh_locks."""
    from superlocalmemory.server import unified_daemon

    tree = ast.parse(inspect.getsource(unified_daemon))
    assigned: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "_mesh_db":
                assigned["_mesh_db"] = ast.unparse(node.value)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "expire_stale_mesh_locks"
    ]
    assert len(calls) == 1, "expected exactly one boot call"
    arg = ast.unparse(calls[0].args[0])
    return assigned.get(arg, arg)


def test_boot_uses_the_store_resolved_from_config() -> None:
    target = _boot_mesh_expiry_target()
    assert "_memory_db" in target, target
    assert not target.startswith("canonical_data_root()"), target
