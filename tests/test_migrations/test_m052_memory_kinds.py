# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M052 adds five empty columns and two tables. It must not touch a memory.

Every test builds a store with rows, a gap in the rowids, the external-content
search index and its triggers, and the two scene triggers that name the table
from elsewhere; then runs the migration and checks that the same memories come
out the other side, findable the same way.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

import pytest

from superlocalmemory.storage.migrations import M052_memory_kinds as M052

from ._kind_store import (
    KIND_COLUMNS,
    FailOnNthAlter,
    columns,
    connect,
    fts_ok,
    object_names,
    reduced_store,
    table_names,
)

_NEW_TABLES = {"memory_kind_runs", "memory_kind_history"}


@pytest.fixture
def store(tmp_path):
    conn = reduced_store(tmp_path / "memory.db")
    yield conn
    conn.close()


def _snapshot(conn) -> list[tuple]:
    return sorted(conn.execute(
        "SELECT rowid, fact_id, profile_id, content, fact_type FROM atomic_facts"
    ).fetchall())


def _match(conn, term: str) -> list[str]:
    return sorted(r[0] for r in conn.execute(
        "SELECT af.fact_id FROM atomic_facts_fts "
        "JOIN atomic_facts AS af ON af.rowid = atomic_facts_fts.rowid "
        "WHERE atomic_facts_fts MATCH ?", (term,)))


def test_apply_adds_five_nullable_columns_without_touching_rows(store) -> None:
    before = _snapshot(store)

    M052.apply(store)

    assert _snapshot(store) == before
    info = {str(r[1]): r for r in store.execute("PRAGMA table_info(atomic_facts)")}
    for name in KIND_COLUMNS:
        assert name in info, name
        assert info[name][3] == 0, f"{name} must be nullable"
        assert info[name][4] is None, f"{name} must default to NULL"
    assert store.execute(
        "SELECT COUNT(*) FROM atomic_facts WHERE memory_kind IS NOT NULL "
        "OR memory_kind_source IS NOT NULL OR memory_kind_confidence IS NOT NULL "
        "OR memory_kind_recipe IS NOT NULL OR memory_kind_at IS NOT NULL"
    ).fetchone()[0] == 0
    assert "idx_facts_memory_kind" in object_names(store, "index")
    assert _NEW_TABLES <= table_names(store)


def test_fts_still_maps_rowids_to_the_same_facts(store) -> None:
    terms = ("pipeline", "market", "Paris", "review")
    before = {t: _match(store, t) for t in terms}
    assert all(before.values()), "the fixture must make every term findable"

    M052.apply(store)

    assert {t: _match(store, t) for t in terms} == before
    assert fts_ok(store)


def test_triggers_survive_and_index_new_writes(store) -> None:
    triggers_before = object_names(store, "trigger")
    assert len(triggers_before) == 5

    M052.apply(store)

    assert object_names(store, "trigger") == triggers_before
    store.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, content, fact_type, "
        "memory_kind, memory_kind_source) VALUES "
        "('n1','m9','Always run the linter before committing','semantic','rule','caller')"
    )
    assert _match(store, "linter") == ["n1"]
    store.execute("INSERT INTO memory_scenes VALUES ('s2','default','[\"n1\"]')")
    assert store.execute(
        "SELECT fact_id FROM scene_fact_members WHERE scene_id='s2'"
    ).fetchall() == [("n1",)]
    assert fts_ok(store)


def test_apply_twice_is_a_no_op_and_verify_holds(store) -> None:
    M052.apply(store)
    schema_once = sorted(store.execute(
        "SELECT type, name, sql FROM sqlite_master").fetchall())
    rows_once = _snapshot(store)

    M052.apply(store)
    M052.repair(store)

    assert sorted(store.execute(
        "SELECT type, name, sql FROM sqlite_master").fetchall()) == schema_once
    assert _snapshot(store) == rows_once
    assert M052.verify(store) is True
    assert M052.unmet(store) == ""
    assert M052.blocks_serving(store) is False


def test_verify_names_what_is_missing(store) -> None:
    assert M052.verify(store) is False
    assert "memory_kind_runs" in M052.unmet(store)
    M052.apply(store)
    store.execute("DROP INDEX idx_facts_memory_kind")
    assert M052.verify(store) is False
    assert "idx_facts_memory_kind" in M052.unmet(store)


def test_fresh_store_without_atomic_facts_creates_tables_only(tmp_path) -> None:
    conn = connect(tmp_path / "memory.db")
    try:
        M052.apply(conn)
        assert _NEW_TABLES <= table_names(conn)
        assert "atomic_facts" not in table_names(conn)
        assert M052.verify(conn) is True
    finally:
        conn.close()


@pytest.mark.parametrize("nth", [1, 3, 5])
def test_failure_mid_apply_rolls_back_everything(tmp_path, nth) -> None:
    db = tmp_path / "memory.db"
    raw = reduced_store(db)
    before_rows = _snapshot(raw)
    before_schema = sorted(raw.execute("SELECT type, name, sql FROM sqlite_master"))

    with pytest.raises(sqlite3.OperationalError, match="injected"):
        M052.apply(FailOnNthAlter(raw, nth))
    assert not raw.in_transaction, "the failed apply left a transaction open"
    raw.close()

    check = connect(db)
    try:
        assert not set(KIND_COLUMNS) & set(columns(check)), "a column survived"
        assert not _NEW_TABLES & table_names(check), "a new table survived"
        assert sorted(check.execute(
            "SELECT type, name, sql FROM sqlite_master")) == before_schema
        assert _snapshot(check) == before_rows
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        check.close()


def test_a_row_count_change_inside_the_transaction_rolls_back(store, monkeypatch) -> None:
    """The in-transaction count guard is live, not decorative."""
    real_count = M052._count
    calls = {"n": 0}

    def drifting(conn, sql):
        calls["n"] += 1
        value = real_count(conn, sql)
        return value + 1 if calls["n"] == 2 else value

    monkeypatch.setattr(M052, "_count", drifting)
    with pytest.raises(sqlite3.OperationalError, match="row count"):
        M052.apply(store)
    assert not set(KIND_COLUMNS) & set(columns(store))
    assert not _NEW_TABLES & table_names(store)


def test_the_module_declares_an_additive_migration() -> None:
    assert M052.NAME == "M052_memory_kinds"
    assert M052.DB_TARGET == "memory"
    assert M052.BREAKING_VERSION == 0
    assert M052.DOWNGRADE_FLOOR == 51
    assert "CHECK" not in M052._IDX_FACTS_MEMORY_KIND
    for name, sql_type in M052._COLUMNS:
        assert name in KIND_COLUMNS and sql_type in {"TEXT", "REAL"}


def test_no_constraint_means_no_kind_value_can_fail_a_write(store) -> None:
    """Invariant I1 at the storage layer: an unexpected value is stored, not refused."""
    M052.apply(store)
    store.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, content, memory_kind, "
        "memory_kind_confidence) VALUES ('odd','m9','x','not-a-kind', 7.5)"
    )
    assert store.execute(
        "SELECT memory_kind FROM atomic_facts WHERE fact_id='odd'"
    ).fetchone() == ("not-a-kind",)


def test_new_tables_carry_profile_id_for_export_and_erasure(store) -> None:
    M052.apply(store)
    for table in _NEW_TABLES:
        assert "profile_id" in columns(store, table), table
    store.execute(
        "INSERT INTO memory_kind_runs (run_id, profile_id, status, backend, recipe_id, "
        "mode, requested_by, created_at, updated_at) VALUES "
        "('r1','default','running','rules','kinds-v1','untyped','cli','t','t')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO memory_kind_runs (run_id, profile_id, status, backend, "
            "recipe_id, mode, requested_by, created_at, updated_at) VALUES "
            "('r2','default','queued','rules','kinds-v1','untyped','cli','t','t')"
        )


# ---------------------------------------------------------------------------
# G5: a copy of a real store. Never the live store and never the backup itself.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("SLM_REAL_SNAPSHOT"),
    reason="set SLM_REAL_SNAPSHOT to a COPY of a real memory.db",
)
def test_real_store_copy(tmp_path, monkeypatch, capsys) -> None:
    from superlocalmemory.storage import migration_runner as mr
    from superlocalmemory.storage._schema_version import read_schema_version

    source = Path(os.environ["SLM_REAL_SNAPSHOT"])
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    shutil.copy2(source, memory_db)
    if (source.parent / "learning.db").is_file():
        shutil.copy2(source.parent / "learning.db", learning_db)

    def type_counts() -> Counter:
        conn = sqlite3.connect(f"file:{memory_db}?mode=ro", uri=True)
        try:
            return Counter(dict(conn.execute(
                "SELECT fact_type, COUNT(*) FROM atomic_facts GROUP BY fact_type")))
        finally:
            conn.close()

    before = type_counts()
    timings: list[float] = []
    real_apply = M052.apply

    def timed(conn):
        start = time.perf_counter()
        try:
            real_apply(conn)
        finally:
            timings.append(time.perf_counter() - start)

    monkeypatch.setattr(M052, "apply", timed)
    eager = mr.apply_all(learning_db, memory_db)
    deferred = mr.apply_deferred(learning_db, memory_db)

    assert "M052_memory_kinds" in eager["applied"], eager["details"].get("M052_memory_kinds")
    assert eager["failed"] == [] and deferred["failed"] == [], (eager, deferred)
    assert len(timings) == 1 and timings[0] < 2.0, timings
    after = type_counts()
    assert after == before
    conn = connect(memory_db)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        assert integrity == "ok"
        assert fts_ok(conn)
        assert M052.verify(conn) is True
    finally:
        conn.close()
    assert read_schema_version(memory_db) == 52
    with capsys.disabled():
        print(f"\nG5 M052 apply seconds={timings[0]:.3f} "
              f"fact_type before={dict(before)} after={dict(after)} "
              f"integrity={integrity} fts=ok stamp={read_schema_version(memory_db)}")
