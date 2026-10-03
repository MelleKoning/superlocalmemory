# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for DatabaseManager's memory-kind column support (WP-2, M052).

M052 itself (the migration that adds these columns and tables, WP-1) has not
landed in this tree yet. Per the WP-2 brief, the fixture below creates the
five ``atomic_facts`` columns and the two kind tables itself, exactly per
LLD §4.2's DDL, so these tests are not blocked on WP-1 and exercise the real
code path a migrated store will use.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

_KIND_COLUMN_DDL = (
    ("memory_kind", "TEXT"),
    ("memory_kind_source", "TEXT"),
    ("memory_kind_confidence", "REAL"),
    ("memory_kind_recipe", "TEXT"),
    ("memory_kind_at", "TEXT"),
)

_RUNS_TABLE_DDL = """
CREATE TABLE memory_kind_runs (
    run_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued','running','paused','completed',
                                           'cancelled','failed','reverting','reverted')),
    backend TEXT NOT NULL, recipe_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('untyped','refresh')),
    cursor_rowid INTEGER NOT NULL DEFAULT 0, revert_cursor INTEGER,
    total_estimate INTEGER NOT NULL DEFAULT 0, processed INTEGER NOT NULL DEFAULT 0,
    changed INTEGER NOT NULL DEFAULT 0, skipped INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0, last_error TEXT,
    requested_by TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT,
    finished_at TEXT, updated_at TEXT NOT NULL)
"""

_HISTORY_TABLE_DDL = """
CREATE TABLE memory_kind_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL, profile_id TEXT NOT NULL, run_id TEXT,
    origin TEXT NOT NULL CHECK (origin IN ('backfill','user_edit','revert','reconcile','restore')),
    old_kind TEXT, old_source TEXT, old_confidence REAL, old_fact_type TEXT,
    new_kind TEXT, new_source TEXT, new_confidence REAL, new_fact_type TEXT,
    actor TEXT NOT NULL, changed_at TEXT NOT NULL)
"""


def _apply_memory_kind_schema(db: DatabaseManager) -> None:
    """Upgrade the store with the real M052 migration."""
    from superlocalmemory.storage.migrations import M052_memory_kinds as m052

    with db.raw_connection() as conn:
        m052.apply(conn)
    db._kind_columns_present = True  # noqa: SLF001 - test shortcut, mirrors a fresh probe


def _strip_kind_columns(db: DatabaseManager) -> None:
    """Make a fresh store look like one from before M052: no kind columns,
    index or tables (the current schema creates them)."""
    with db.raw_connection() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_facts_memory_kind")
        present = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
        for name, _sql_type in _KIND_COLUMN_DDL:
            if name in present:
                conn.execute(f"ALTER TABLE atomic_facts DROP COLUMN {name}")
        conn.execute("DROP TABLE IF EXISTS memory_kind_history")
        conn.execute("DROP TABLE IF EXISTS memory_kind_runs")
    db._kind_columns_present = False  # noqa: SLF001 - forget any earlier probe
    db._kind_columns_checked_at = 0.0  # noqa: SLF001


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    _strip_kind_columns(mgr)
    return mgr


@pytest.fixture()
def kind_db(db: DatabaseManager) -> DatabaseManager:
    _apply_memory_kind_schema(db)
    return db


def _fact(fact_id: str, memory_id: str, content: str, *, kind: str | None,
          source: str | None, confidence: float | None = None) -> AtomicFact:
    f = AtomicFact(
        fact_id=fact_id, profile_id="default", memory_id=memory_id,
        content=content, fact_type=FactType.SEMANTIC,
    )
    f.memory_kind = kind
    f.memory_kind_source = source
    f.memory_kind_confidence = confidence
    f.memory_kind_recipe = "test" if kind else None
    f.memory_kind_at = "2026-01-01T00:00:00+00:00" if kind else None
    return f


class TestPreM052:
    def test_pre_m052_table_still_accepts_writes(self, db: DatabaseManager) -> None:
        assert db.has_memory_kind_columns() is False
        mid = db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        fid = db.store_fact(_fact("f1", mid, "hello world", kind=None, source=None))
        facts = db.get_all_facts("default")
        assert len(facts) == 1
        assert facts[0].fact_id == fid
        assert facts[0].memory_kind is None


class TestKindColumnsPresent:
    def test_kind_columns_written_when_present(self, kind_db: DatabaseManager) -> None:
        assert kind_db.has_memory_kind_columns() is True
        mid = kind_db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        kind_db.store_fact(
            _fact("f1", mid, "always say please", kind="rule", source="caller"),
        )
        row = kind_db.get_all_facts("default")[0]
        assert row.memory_kind == "rule"
        assert row.memory_kind_source == "caller"
        assert row.fact_type == FactType.SEMANTIC

    def test_upsert_user_kind_beats_model_suggestion(self, kind_db: DatabaseManager) -> None:
        mid = kind_db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        kind_db.store_fact(
            _fact("f1", mid, "content A", kind="procedure", source="user"),
        )
        kind_db.store_fact(
            _fact("f1", mid, "content B", kind="opinion", source="model:llm"),
        )
        row = kind_db.get_all_facts("default")[0]
        assert row.content == "content B"  # non-kind columns still update
        assert row.memory_kind == "procedure"
        assert row.memory_kind_source == "user"

    def test_upsert_caller_over_caller_takes_new(self, kind_db: DatabaseManager) -> None:
        mid = kind_db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        kind_db.store_fact(
            _fact("f1", mid, "content A", kind="rule", source="caller"),
        )
        kind_db.store_fact(
            _fact("f1", mid, "content B", kind="decision", source="caller"),
        )
        row = kind_db.get_all_facts("default")[0]
        assert row.memory_kind == "decision"
        assert row.memory_kind_source == "caller"

    def test_upsert_keeps_fact_type_for_confirmed_rows(self, kind_db: DatabaseManager) -> None:
        mid = kind_db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        original = _fact("f1", mid, "content A", kind="rule", source="caller")
        original.fact_type = FactType.SEMANTIC
        kind_db.store_fact(original)
        # A lower-authority write tries to change fact_type too; it must not
        # take, because the kind it is attached to does not take either.
        challenger = _fact("f1", mid, "content B", kind="episodic", source="model:llm")
        challenger.fact_type = FactType.EPISODIC
        kind_db.store_fact(challenger)
        row = kind_db.get_all_facts("default")[0]
        assert row.memory_kind == "rule"
        assert row.fact_type == FactType.SEMANTIC

    def test_hydration_reads_kind_fields(self, kind_db: DatabaseManager) -> None:
        mid = kind_db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        kind_db.store_fact(
            _fact("f1", mid, "ship on Tuesday", kind="decision", source="caller", confidence=None),
        )
        row = kind_db.get_all_facts("default")[0]
        assert row.memory_kind == "decision"
        assert row.memory_kind_source == "caller"
        assert row.memory_kind_recipe == "test"
        assert row.memory_kind_at == "2026-01-01T00:00:00+00:00"

    def test_old_style_hydration_ignores_new_columns(self, db: DatabaseManager) -> None:
        """A fact stored before M052 hydrates cleanly once the columns exist."""
        mid = db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        db.store_fact(_fact("f1", mid, "pre-existing", kind=None, source=None))
        _apply_memory_kind_schema(db)
        row = db.get_all_facts("default")[0]
        assert row.memory_kind is None
        assert row.content == "pre-existing"

    def test_update_fact_accepts_kind_columns(self, kind_db: DatabaseManager) -> None:
        mid = kind_db.store_memory(MemoryRecord(profile_id="default", content="hi"))
        fid = kind_db.store_fact(_fact("f1", mid, "note this", kind=None, source=None))
        kind_db.update_fact(fid, {
            "memory_kind": "status", "memory_kind_source": "user",
            "memory_kind_confidence": None, "memory_kind_recipe": "manual",
            "memory_kind_at": "2026-02-02T00:00:00+00:00",
        })
        row = kind_db.get_all_facts("default")[0]
        assert row.memory_kind == "status"
        assert row.memory_kind_source == "user"
        assert row.memory_kind_recipe == "manual"


class TestCapabilityProbe:
    def test_capability_probe_rechecks_false_after_5s(
        self, db: DatabaseManager, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

        assert db.has_memory_kind_columns() is False
        # Within the 5s window: no re-probe, still False, without touching SQL.
        clock["t"] += 1.0
        assert db.has_memory_kind_columns() is False

        # Add the columns "out of band" and advance the clock past 5s: the
        # next call must re-probe and flip to True (and then stay True).
        with db.raw_connection() as conn:
            for name, sql_type in _KIND_COLUMN_DDL:
                conn.execute(f"ALTER TABLE atomic_facts ADD COLUMN {name} {sql_type}")
            conn.execute(_RUNS_TABLE_DDL)
            conn.execute(_HISTORY_TABLE_DDL)
        clock["t"] += 5.1
        assert db.has_memory_kind_columns() is True
        clock["t"] += 100
        assert db.has_memory_kind_columns() is True
