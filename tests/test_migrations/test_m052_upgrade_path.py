# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M052 inside the real upgrade path: registration, stamp, retries, old builds.

The unit tests prove the migration itself. These prove what a user upgrading
with pip or npm actually runs into: the runner picks it up eagerly, a crash
between COMMIT and the log write is retried to completion, the engine can still
start on a store the migration never reached, an older build refuses the
stamped store, and an older writer still running beside it keeps working.
"""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import pytest

import superlocalmemory.storage._schema_version as sv
import superlocalmemory.storage.migration_runner as mr
from superlocalmemory.storage import schema
from superlocalmemory.storage._migration_internals import (
    _MODULES,
    _apply_single,
    _ensure_migration_log,
    _get_log_row,
)
from superlocalmemory.storage.migrations import M052_memory_kinds as M052

from ._kind_store import KIND_COLUMNS, columns, connect, full_pre_kind_store, object_names

_HOLD_MODULE = "superlocalmemory.storage.upgrade_restore"


def _migration():
    return next(m for m in mr.MIGRATIONS if m.name == M052.NAME)


def _fresh_engine_store(tmp_path: Path) -> tuple[Path, Path]:
    memory_db, learning_db = tmp_path / "memory.db", tmp_path / "learning.db"
    with sqlite3.connect(memory_db) as conn:
        schema.create_all_tables(conn)
    return learning_db, memory_db


def test_m052_is_eager_and_registered() -> None:
    eager = [m.name for m in mr.MIGRATIONS]
    assert M052.NAME in eager
    assert M052.NAME not in [m.name for m in mr.DEFERRED_MIGRATIONS]
    assert _MODULES[M052.NAME] is M052
    assert _migration().db_target == "memory"
    assert _migration().ddl == M052.DDL
    assert mr._M052 is M052


def test_completion_certificate_stamps_52(tmp_path) -> None:
    learning_db, memory_db = _fresh_engine_store(tmp_path)
    eager = mr.apply_all(learning_db, memory_db)
    assert eager["failed"] == [], eager["details"]
    deferred = mr.apply_deferred(learning_db, memory_db)
    assert deferred["failed"] == [], deferred["details"]
    assert sv.SUPPORTED_SCHEMA_VERSION == 52
    assert sv.read_schema_version(memory_db) == 52
    assert sv.read_schema_version(learning_db) == 52
    assert mr.status(learning_db, memory_db)[M052.NAME] == "complete"


def test_a_51_ceiling_build_refuses_a_52_store(tmp_path, monkeypatch) -> None:
    learning_db, memory_db = _fresh_engine_store(tmp_path)
    mr.apply_all(learning_db, memory_db)
    mr.apply_deferred(learning_db, memory_db)
    monkeypatch.setattr(sv, "SUPPORTED_SCHEMA_VERSION", 51)
    monkeypatch.setattr(sv, "_detect_all_installs", lambda: [])
    with pytest.raises(sv.SchemaVersionError, match="too old"):
        sv.check_version_or_raise(memory_db)


def test_in_progress_log_after_commit_is_retried_to_complete(tmp_path) -> None:
    """Power lost after COMMIT but before the log said 'complete'."""
    conn = full_pre_kind_store(tmp_path / "memory.db")
    try:
        _ensure_migration_log(conn)
        M052.apply(conn)
        conn.execute(
            "INSERT INTO migration_log (name, applied_at, ddl_sha256, rows_affected, "
            "status) VALUES (?, 't', 'x', 0, 'in_progress')", (M052.NAME,))

        outcome, detail = _apply_single(conn, _migration(), dry_run=False)

        assert outcome == "applied", detail
        assert _get_log_row(conn, M052.NAME)[4] == "complete"
        assert M052.verify(conn) is True
    finally:
        conn.close()


def test_engine_still_starts_on_a_store_m052_never_reached(tmp_path) -> None:
    """Disk too small for the snapshot: apply_all never ran. The engine must start
    and every write must keep working, with the kind columns honestly absent."""
    conn = full_pre_kind_store(tmp_path / "memory.db")
    try:
        schema.create_all_tables(conn)  # the next engine start
        assert not set(KIND_COLUMNS) & set(columns(conn))
        conn.execute(
            "INSERT INTO memories (memory_id, content) VALUES ('m1', 'hello')")
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, content) "
            "VALUES ('f1', 'm1', 'hello')")
        M052.apply(conn)
        schema.create_all_tables(conn)
        assert M052.verify(conn) is True
    finally:
        conn.close()


def test_fresh_install_converges_to_the_same_schema_as_an_upgrade(tmp_path) -> None:
    fresh_dir, upgraded_dir = tmp_path / "fresh", tmp_path / "upgraded"
    fresh_dir.mkdir()
    upgraded_dir.mkdir()
    fresh = connect(fresh_dir / "memory.db")
    M052.apply(fresh)                 # apply_all runs before the engine
    schema.create_all_tables(fresh)   # then the engine creates atomic_facts
    upgraded = full_pre_kind_store(upgraded_dir / "memory.db")
    M052.apply(upgraded)
    try:
        assert M052.verify(fresh) is True and M052.verify(upgraded) is True
        for conn in (fresh, upgraded):
            info = {str(r[1]): (r[2], r[3], r[4])
                    for r in conn.execute("PRAGMA table_info(atomic_facts)")}
            assert {c: info[c] for c in KIND_COLUMNS} == {
                c: ("REAL" if c == "memory_kind_confidence" else "TEXT", 0, None)
                for c in KIND_COLUMNS}

        def idx(conn):
            return " ".join(conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='idx_facts_memory_kind'"
            ).fetchone()[0].split())
        assert idx(fresh) == idx(upgraded)
    finally:
        fresh.close()
        upgraded.close()


def test_a_later_table_rebuild_keeps_the_kind_columns_and_index(tmp_path) -> None:
    """Upgrading from before 4.1.14: eager M052 runs first, then deferred M046
    rebuilds atomic_facts. The rebuild must carry the new columns and index."""
    from superlocalmemory.storage.migrations import (
        M046_prospective_memory_has_its_own_name as M046,
    )

    conn = connect(tmp_path / "memory.db")
    try:
        conn.executescript(
            "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, memory_id TEXT, "
            "profile_id TEXT NOT NULL DEFAULT 'default', content TEXT, "
            "fact_type TEXT NOT NULL DEFAULT 'semantic' CHECK (fact_type IN "
            "('episodic','semantic','opinion','temporal')));"
            "INSERT INTO atomic_facts VALUES ('f1','m1','default','x','temporal');"
        )
        M052.apply(conn)
        conn.execute(
            "UPDATE atomic_facts SET memory_kind='prospective', "
            "memory_kind_source='user' WHERE fact_id='f1'")
        M046.apply(conn)
        assert M046.verify(conn) is True and M052.verify(conn) is True
        assert conn.execute(
            "SELECT fact_type, memory_kind, memory_kind_source FROM atomic_facts"
        ).fetchone() == ("prospective", "prospective", "user")
        assert "idx_facts_memory_kind" in object_names(conn, "index")
    finally:
        conn.close()


def test_old_style_hydration_ignores_new_columns(tmp_path) -> None:
    """The 4.1.18 reader and writer on a migrated store: SELECT * gains five
    columns it never names, and its explicit-column writes still land."""
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

    conn = full_pre_kind_store(tmp_path / "memory.db")
    M052.apply(conn)
    conn.close()

    db = DatabaseManager(tmp_path / "memory.db")
    db.store_memory(MemoryRecord(memory_id="m1", content="hello world"))
    db.store_fact(AtomicFact(fact_id="f1", memory_id="m1", content="hello world",
                             fact_type=FactType.SEMANTIC))
    with sqlite3.connect(tmp_path / "memory.db") as raw:
        raw.execute(
            "UPDATE atomic_facts SET memory_kind='rule', memory_kind_source='caller' "
            "WHERE fact_id='f1'")
    fact = db.get_fact("f1")
    assert fact is not None and fact.content == "hello world"
    assert fact.fact_type is FactType.SEMANTIC


# ---------------------------------------------------------------------------
# The prepare-for-downgrade hold (WP-6 owns the predicate)
# ---------------------------------------------------------------------------

def _install_hold(monkeypatch, fn) -> None:
    module = types.ModuleType(_HOLD_MODULE)
    module.downgrade_hold_active = fn
    monkeypatch.setitem(sys.modules, _HOLD_MODULE, module)


def test_runner_stamps_normally_when_the_hold_module_is_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, _HOLD_MODULE, None)  # import raises ImportError
    learning_db, memory_db = _fresh_engine_store(tmp_path)
    mr.apply_all(learning_db, memory_db)
    result = mr.apply_deferred(learning_db, memory_db)
    assert result["failed"] == [], result["details"]
    assert sv.read_schema_version(memory_db) == 52


def test_runner_holds_the_stamp_while_a_downgrade_is_prepared(tmp_path, monkeypatch) -> None:
    seen: list[Path] = []
    _install_hold(monkeypatch, lambda root: seen.append(Path(root)) or True)
    learning_db, memory_db = _fresh_engine_store(tmp_path)
    mr.apply_all(learning_db, memory_db)
    result = mr.apply_deferred(learning_db, memory_db)
    assert result["failed"] == [], result["details"]
    assert sv.read_schema_version(memory_db) < 52
    assert sv.read_schema_version(learning_db) < 52
    assert seen and seen[0] == memory_db.parent
    assert "held" in result["details"]["schema_version_stamp"]


def test_a_broken_hold_predicate_stamps_normally(tmp_path, monkeypatch) -> None:
    """Fail toward the guard: an older build must still be refused."""
    def boom(_root):
        raise RuntimeError("marker unreadable")
    _install_hold(monkeypatch, boom)
    learning_db, memory_db = _fresh_engine_store(tmp_path)
    mr.apply_all(learning_db, memory_db)
    result = mr.apply_deferred(learning_db, memory_db)
    assert result["failed"] == [], result["details"]
    assert sv.read_schema_version(memory_db) == 52
