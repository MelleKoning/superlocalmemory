# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The login policy is read from the store only when the store has changed.

Every request asked ``RbacEngine.require_login()``, and every ask opened a new
database connection, on the service's request loop. When opening a connection
stalled (seconds, while a large write was committing), every request in flight
froze with it: a recall of 1.1 s retrieval answered after 5.1 s. The value is
now kept while nothing has been committed to the store (storage/store_signature),
so a change made anywhere, by any process, is still seen on the next request.
"""

from __future__ import annotations

from superlocalmemory.access import rbac as rbac_mod
from superlocalmemory.access.rbac import RbacEngine
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager


def _store(tmp_path):
    import sqlite3

    from superlocalmemory.storage.migrations import M024_rbac_users_roles as m024

    path = tmp_path / "memory.db"
    DatabaseManager(path).initialize(schema)
    conn = sqlite3.connect(path)
    m024.apply(conn)
    conn.commit()
    conn.close()
    return path


def _count_connections(engine: RbacEngine, monkeypatch) -> list[int]:
    opened = [0]
    real = engine._conn

    def counting():
        opened[0] += 1
        return real()

    monkeypatch.setattr(engine, "_conn", counting)
    return opened


def test_an_unchanged_policy_opens_no_connection(tmp_path, monkeypatch) -> None:
    engine = RbacEngine(_store(tmp_path))
    opened = _count_connections(engine, monkeypatch)
    assert engine.require_login() is False
    first = opened[0]
    for _ in range(20):
        assert engine.require_login() is False
    assert opened[0] == first


def test_a_change_by_another_process_is_seen_at_once(tmp_path) -> None:
    path = _store(tmp_path)
    reader, writer = RbacEngine(path), RbacEngine(path)
    assert reader.require_login() is False
    writer.set_require_login(True)              # another engine, as another process
    assert reader.require_login() is True
    writer.set_require_login(False)
    assert reader.require_login() is False


def test_an_unreadable_signature_reads_the_store(tmp_path, monkeypatch) -> None:
    engine = RbacEngine(_store(tmp_path))
    monkeypatch.setattr(rbac_mod.store_signature, "of", lambda _path: None, raising=False)
    opened = _count_connections(engine, monkeypatch)
    engine.require_login()
    engine.require_login()
    assert opened[0] == 2
