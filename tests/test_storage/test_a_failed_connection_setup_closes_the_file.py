# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A connection whose setup fails is closed at once, not when the error dies.

Every SLM connection helper opens a database and then configures it. When a
configuring statement raised ("database is locked", a damaged file), the
connection was left open for as long as the exception lived -- a log record
keeps it -- and on Windows an open database cannot be deleted or replaced.
The CI guard (tests/test_ci_guards/test_sqlite_connections_are_closed.py)
checks the shape everywhere; these run three of the helpers for real.
"""

from __future__ import annotations

import sqlite3

import pytest

_REAL_CONNECT = sqlite3.connect


class _PragmaFails(sqlite3.Connection):
    """A connection whose first PRAGMA raises, as a locked database does."""

    opened: list["_PragmaFails"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _PragmaFails.opened.append(self)

    def execute(self, sql, *args):
        if str(sql).lstrip().upper().startswith("PRAGMA"):
            raise sqlite3.OperationalError("database is locked")
        return super().execute(sql, *args)


def _is_closed(conn: sqlite3.Connection) -> bool:
    try:
        sqlite3.Connection.execute(conn, "SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


@pytest.fixture()
def failing_setup(monkeypatch):
    _PragmaFails.opened.clear()

    def connect(*args, **kwargs):
        kwargs["factory"] = _PragmaFails
        return _REAL_CONNECT(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    return _PragmaFails.opened


def test_the_access_control_store(failing_setup, tmp_path):
    from superlocalmemory.access.rbac import RbacEngine

    engine = RbacEngine.__new__(RbacEngine)
    engine._db_path = str(tmp_path / "rbac.db")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        engine._conn()
    assert failing_setup and all(_is_closed(c) for c in failing_setup)


def test_the_code_graph_store(failing_setup, tmp_path):
    from superlocalmemory.code_graph.database import CodeGraphDatabase

    graph = CodeGraphDatabase.__new__(CodeGraphDatabase)
    graph.db_path = tmp_path / "graph.db"
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        graph._connect()
    assert failing_setup and all(_is_closed(c) for c in failing_setup)


def test_a_cached_connection_is_not_kept_after_a_failed_setup(failing_setup, tmp_path):
    """The reward model caches its connection: a failed setup must neither
    stay open nor stay cached, or every later call would reuse it."""
    from superlocalmemory.learning.reward import EngagementRewardModel

    model = EngagementRewardModel.__new__(EngagementRewardModel)
    model._db = tmp_path / "memory.db"
    model._conn = None
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        model._get_conn()
    assert model._conn is None
    assert failing_setup and all(_is_closed(c) for c in failing_setup)
