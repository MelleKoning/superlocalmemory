# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M053: the Answer Check history tables, additive, in learning.db."""

from __future__ import annotations

import sqlite3

import superlocalmemory.storage._schema_version as sv
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage._downgrade import _blockers
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migrations import M053_answer_check_history as M053


def _stores(tmp_path):
    from superlocalmemory.storage import schema

    learning, memory = tmp_path / "learning.db", tmp_path / "memory.db"
    with sqlite3.connect(memory) as conn:
        schema.create_all_tables(conn)
    return learning, memory


def test_registered_eager_on_learning() -> None:
    m = next(m for m in mr.MIGRATIONS if m.name == M053.NAME)
    assert m.db_target == "learning" == M053.DB_TARGET
    assert _MODULES[M053.NAME] is M053
    assert sv.SUPPORTED_SCHEMA_VERSION == 53


def test_applies_on_fresh_store_and_is_idempotent(tmp_path) -> None:
    learning, memory = _stores(tmp_path)
    first = mr.apply_all(learning, memory)
    assert M053.NAME in first["applied"] and first["failed"] == []
    second = mr.apply_all(learning, memory)
    assert M053.NAME not in second["applied"] and second["failed"] == []
    with sqlite3.connect(learning) as conn:
        assert M053.verify(conn) is True
        cols = [r[1] for r in conn.execute("PRAGMA table_info(answer_check_events)")]
    assert not {"query", "query_text", "content", "fact_id", "fact_ids", "session_id",
                "agent_id"} & set(cols)


def test_applies_on_a_4_1_19_shaped_store(tmp_path) -> None:
    learning, memory = _stores(tmp_path)
    mr.apply_all(learning, memory)
    with sqlite3.connect(learning) as conn:          # what 4.1.19 left behind
        conn.execute("DROP TABLE answer_check_events")
        conn.execute("DROP TABLE answer_check_erasures")
        conn.execute("DELETE FROM migration_log WHERE name = ?", (M053.NAME,))
        assert M053.verify(conn) is False
    result = mr.apply_all(learning, memory)
    assert M053.NAME in result["applied"], result
    with sqlite3.connect(learning) as conn:
        assert M053.verify(conn) is True


def test_repair_restores_a_dropped_index(tmp_path) -> None:
    learning, memory = _stores(tmp_path)
    mr.apply_all(learning, memory)
    with sqlite3.connect(learning) as conn:
        conn.execute("DROP INDEX idx_answer_check_events_profile_time")
        assert M053.verify(conn) is False
        M053.repair(conn)
        assert M053.verify(conn) is True


def test_downgrade_to_4_1_19_and_4_1_18_not_blocked(tmp_path) -> None:
    learning, memory = _stores(tmp_path)
    mr.apply_all(learning, memory)
    mr.apply_deferred(learning, memory)
    assert not [p for p in _blockers(learning, memory, 52) if M053.NAME in p]
    assert not [p for p in _blockers(learning, memory, 51) if M053.NAME in p]
    assert M053.BREAKING_VERSION == 0
