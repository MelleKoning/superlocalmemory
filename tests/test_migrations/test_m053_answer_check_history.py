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
    assert sv.SUPPORTED_SCHEMA_VERSION >= 53  # 54 since M054 (saved views)


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


#: The unreleased development M053, before the stage-timing columns. No released
#: build ran M053 (v4.1.19 has none); a 4.1.20 development store might have.
_DEV_EVENTS_DDL = """
CREATE TABLE answer_check_events (
    event_id TEXT NOT NULL, profile_id TEXT NOT NULL, occurred_ms INTEGER NOT NULL,
    status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '', backend TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT '', abstained INTEGER NOT NULL DEFAULT 0,
    abstention_reason TEXT, answer_confidence REAL, threshold REAL,
    reordered INTEGER NOT NULL DEFAULT 0, result_count INTEGER NOT NULL DEFAULT 0,
    query_type TEXT NOT NULL DEFAULT '', retrieval_ms REAL, judge_ms REAL, total_ms REAL,
    calibration_id TEXT NOT NULL DEFAULT '', PRIMARY KEY (profile_id, event_id));
CREATE INDEX idx_answer_check_events_profile_time
    ON answer_check_events (profile_id, occurred_ms DESC);
"""
_DEV_HASH = "3ff0b2d15dd73eb00ffae2347386234f015032e6889391924aa34ed9b8bb89e9"


def test_the_stage_timing_columns_are_part_of_m053(tmp_path) -> None:
    learning, memory = _stores(tmp_path)
    mr.apply_all(learning, memory)
    with sqlite3.connect(learning) as conn:
        cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(answer_check_events)")}
    assert cols["embed_ms"] == "REAL" and cols["rerank_ms"] == "REAL"
    # The stage columns belong to M053 itself; M054 (saved views) adds none.
    from superlocalmemory.storage.migrations import M054_saved_views as M054

    assert "embed_ms" not in M054.DDL and "rerank_ms" not in M054.DDL


def test_a_development_store_without_the_stage_columns_is_repaired(tmp_path) -> None:
    learning, memory = _stores(tmp_path)
    mr.apply_all(learning, memory)
    with sqlite3.connect(learning) as conn:          # what a dev build left behind
        conn.execute("DROP TABLE answer_check_events")
        conn.executescript(_DEV_EVENTS_DDL)
        conn.execute("INSERT INTO answer_check_events (event_id, profile_id, occurred_ms, "
                     "status, retrieval_ms) VALUES ('e1', 'default', 1, 'judged', 700.0)")
        conn.execute("UPDATE migration_log SET ddl_sha256 = ? WHERE name = ?",
                     (_DEV_HASH, M053.NAME))
        assert M053.verify(conn) is False
    result = mr.apply_all(learning, memory)
    assert result["failed"] == [], result
    with sqlite3.connect(learning) as conn:
        assert M053.verify(conn) is True
        row = conn.execute("SELECT retrieval_ms, embed_ms, rerank_ms FROM "
                           "answer_check_events").fetchone()
    assert row == (700.0, None, None)                 # existing rows kept, untouched
    assert mr.apply_all(learning, memory)["failed"] == []
