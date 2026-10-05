# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What the upgrade safety-copy fixes left open.

1. A clock running behind the newest existing copy named the new copy "older",
   and retention (keep the two newest BY NAME) deleted the copy just taken --
   the only one of the store as it was before this start.
2. The eager pass looked for another running daemon only in ``daemon.pid``,
   which the starting daemon has already overwritten with its own pid, so an
   old daemon still writing went unreported there.
3. A copy interrupted by a hard crash left its ``.partial`` staging file (the
   size of the store) in the copies folder for ever.
5. A store whose one-time repair kept being reported failed never got its
   schema-version completion stamp. It does now (checked here, end to end).

Real temp directories and real SQLite only; never the user's data directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from superlocalmemory.storage import _boot_snapshot
from superlocalmemory.storage._schema_version import SUPPORTED_SCHEMA_VERSION
from superlocalmemory.storage.backup import _gc_old_backups
from superlocalmemory.storage.migration_runner import apply_all, apply_deferred
from superlocalmemory.storage.schema import create_all_tables
from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema
from superlocalmemory.storage.schema_v347 import apply_v347_schema
from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
from superlocalmemory.storage.schema_v3411 import apply_v3411_schema

_STAMP = re.compile(r"-(\d{8}-\d{6}-\d{6})(?:-\d+)?-pre-migration\.db$")


def _engine_schema_bootstrap(memory_db: Path) -> None:
    conn = sqlite3.connect(str(memory_db))
    try:
        create_all_tables(conn)
        conn.commit()
    finally:
        conn.close()
    for step in (apply_v343_schema, apply_v346_schema, apply_v347_schema,
                 apply_v3410_schema, apply_v3411_schema):
        step(str(memory_db))


def _start(learning_db: Path, memory_db: Path) -> tuple[dict, dict]:
    eager = apply_all(learning_db, memory_db)
    _engine_schema_bootstrap(memory_db)
    apply_all(learning_db, memory_db)
    return eager, apply_deferred(learning_db, memory_db)


def _forget_migration(db: Path, name: str) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("DELETE FROM migration_log WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()


def _db(path: Path) -> Path:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture()
def store(tmp_path: Path) -> tuple[Path, Path, Path]:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _start(learning_db, memory_db)
    _start(learning_db, memory_db)
    snapshots = tmp_path / "pre-migration-snapshots"
    for child in list(snapshots.iterdir()):
        child.unlink()
    _boot_snapshot.forget()
    return learning_db, memory_db, snapshots


def _stamps(snapshots: Path) -> set[str]:
    return {m.group(1) for p in snapshots.iterdir() if (m := _STAMP.search(p.name))}


# ---------------------------------------------------------------------------
# 1. the copy just taken survives a clock that runs behind
# ---------------------------------------------------------------------------

class TestAClockBehindTheNewestCopy:

    def test_the_copy_just_taken_is_kept(self, store) -> None:
        learning_db, memory_db, snapshots = store
        # Two copies from a clock that ran ahead (or this clock is now behind).
        for stamp in ("20991231-235958-000001", "20991231-235959-000001"):
            _db(snapshots / f"memory-{stamp}-pre-migration.db")
            _db(snapshots / f"learning-{stamp}-pre-migration.db")
        _forget_migration(memory_db, "M042_correction_case_ledger")
        before = _stamps(snapshots)

        eager = apply_all(learning_db, memory_db)

        taken = _stamps(snapshots) - before
        assert eager["details"].get("_backup") == str(snapshots)
        assert taken, "the copy taken by this start was deleted by retention"
        # Still two generations in all: the new one and the newest of the rest.
        assert len(_stamps(snapshots)) == 2
        assert "20991231-235959-000001" in _stamps(snapshots)

    def test_retention_never_removes_a_protected_generation(self, tmp_path) -> None:
        root = tmp_path / "snaps"
        root.mkdir()
        new = _db(root / "memory-20260101-000000-000001-pre-migration.db")
        for stamp in ("20991231-235958-000001", "20991231-235959-000001"):
            _db(root / f"memory-{stamp}-pre-migration.db")
        _gc_old_backups(root, keep=2, protect=frozenset({new.name}))
        assert new.exists()
        assert {p.name for p in root.iterdir()} == {
            new.name, "memory-20991231-235959-000001-pre-migration.db",
        }


# ---------------------------------------------------------------------------
# 2. the eager pass sees an old daemon by its writer lease
# ---------------------------------------------------------------------------

class TestAnotherDaemonByItsLease:

    def test_an_old_daemon_holding_the_lease_is_reported(self, store, caplog) -> None:
        learning_db, memory_db, _snapshots = store
        # The starting daemon has already written its own pid here.
        (memory_db.parent / "daemon.pid").write_text(str(os.getpid()), encoding="utf-8")
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            Path(f"{memory_db}.writer.lock").write_text(json.dumps(
                {"pid": other.pid, "claimed_at_ms": int(time.time() * 1000)},
            ), encoding="utf-8")
            _forget_migration(memory_db, "M042_correction_case_ledger")
            with caplog.at_level(logging.WARNING):
                eager = apply_all(learning_db, memory_db)
        finally:
            other.kill()
            other.wait()
        assert eager["details"].get("_concurrent_daemon_pid") == str(other.pid)
        assert "still running" in caplog.text
        assert _boot_snapshot.claim(learning_db, memory_db) is None

    def test_a_stale_lease_is_not_reported(self, store, caplog) -> None:
        learning_db, memory_db, _snapshots = store
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        Path(f"{memory_db}.writer.lock").write_text(json.dumps({"pid": dead.pid}), encoding="utf-8")
        _forget_migration(memory_db, "M042_correction_case_ledger")
        with caplog.at_level(logging.WARNING):
            eager = apply_all(learning_db, memory_db)
        assert "_concurrent_daemon_pid" not in eager["details"]
        assert "still running" not in caplog.text


# ---------------------------------------------------------------------------
# 3. staging files a hard crash left behind are cleaned
# ---------------------------------------------------------------------------

class TestCrashedStagingFiles:

    def _partial(self, root: Path, stamp: str, age_s: float) -> list[Path]:
        main = root / f"memory-{stamp}-pre-migration.db.partial"
        main.write_bytes(b"\0" * 4096)
        side = [Path(f"{main}-wal"), Path(f"{main}-shm")]
        for p in side:
            p.write_bytes(b"\0" * 64)
        when = time.time() - age_s
        for p in (main, *side):
            os.utime(p, (when, when))
        return [main, *side]

    def test_an_abandoned_partial_copy_is_removed(self, tmp_path) -> None:
        root = tmp_path / "snaps"
        root.mkdir()
        stale = self._partial(root, "20260101-000000-000001", age_s=3 * 3600)
        _gc_old_backups(root, keep=2)
        assert not any(p.exists() for p in stale)

    def test_a_copy_still_being_written_is_left_alone(self, tmp_path) -> None:
        root = tmp_path / "snaps"
        root.mkdir()
        fresh = self._partial(root, "20260101-000000-000002", age_s=5)
        _gc_old_backups(root, keep=2)
        assert all(p.exists() for p in fresh)

    def test_only_staging_files_of_copies_are_touched(self, tmp_path) -> None:
        root = tmp_path / "snaps"
        root.mkdir()
        other = root / "notes.partial"
        other.write_text("mine", encoding="utf-8")
        when = time.time() - 3 * 3600
        os.utime(other, (when, when))
        _gc_old_backups(root, keep=2)
        assert other.exists()


# ---------------------------------------------------------------------------
# 5. a store whose repair kept failing now gets its completion stamp
# ---------------------------------------------------------------------------

def test_a_store_whose_repair_kept_failing_is_stamped_complete(tmp_path) -> None:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _start(learning_db, memory_db)
    conn = sqlite3.connect(str(memory_db))
    try:
        # The shape that failed M043's check on every start: a memory a
        # summary drew on, since forgotten by its owner (score 0).
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                     "VALUES ('mem1', 'default', 'a conversation')")
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " lifecycle, created_at) VALUES ('gone', 'mem1', 'default', 'x',"
            " 'archived', '2026-03-01T00:00:00+00:00')")
        conn.execute(
            "INSERT INTO fact_retention (fact_id, profile_id, retention_score,"
            " lifecycle_zone) VALUES ('gone', 'default', 0.0, 'forgotten')")
        conn.execute(
            "INSERT INTO fact_consolidations (consolidation_id, profile_id,"
            " consolidated_fact_id, source_fact_ids, created_at) VALUES"
            " ('c1', 'default', 'gist', ?, '2026-03-02T00:00:00+00:00')",
            (json.dumps(["gone"]),))
        conn.commit()
    finally:
        conn.close()
    # Never stamped: it had been reported failed on every earlier start.
    for db in (memory_db, learning_db):
        c = sqlite3.connect(str(db))
        c.execute("UPDATE slm_schema_version SET version = 39 WHERE id = 1")
        c.commit()
        c.close()

    _eager, deferred = _start(learning_db, memory_db)

    assert deferred["failed"] == [], deferred["details"]
    for db in (memory_db, learning_db):
        c = sqlite3.connect(str(db))
        try:
            version = c.execute("SELECT version FROM slm_schema_version").fetchone()[0]
        finally:
            c.close()
        assert version == SUPPORTED_SCHEMA_VERSION, db.name
