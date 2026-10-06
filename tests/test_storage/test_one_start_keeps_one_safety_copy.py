# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""One start, one safety copy -- and the previous upgrade's copy survives.

Observed on a scratch copy of a real 4.1.17 store booted by 4.1.18 with a
missing learning.db, and on the owner's own Sep-12 upgrade: the eager pass and
the deferred pass each copied the whole memory.db, seconds apart. Retention
keeps two GENERATIONS, so one start filled both slots and evicted the copy taken
before the previous upgrade -- the one rollback point that predates this start.

The call order these tests mirror is the daemon's own:

    unified_daemon.py  apply_all(...)                  # eager pass, copies
    unified_daemon.py  housekeeping, engine construction
    engine.py          schema bootstrap, apply_all(...), apply_deferred(...)
    unified_daemon.py  writer lease, journal replay    # ordinary writes begin
    unified_daemon.py  apply_deferred(...)             # must copy for itself

Nothing outside the upgrade itself writes between the first two passes, so the
deferred pass reuses the copy. After writes have begun it must take its own.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import migration_runner
from superlocalmemory.storage.migration_runner import apply_all, apply_deferred
from superlocalmemory.storage.schema import create_all_tables
from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema
from superlocalmemory.storage.schema_v347 import apply_v347_schema
from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
from superlocalmemory.storage.schema_v3411 import apply_v3411_schema

_STAMP = re.compile(r"-(\d{8}-\d{6}-\d{6})(?:-\d+)?-pre-migration\.db$")


def _engine_schema_bootstrap(memory_db: Path) -> None:
    """What MemoryEngine._init_db_layer does to the store before its passes."""
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
    """One daemon start, in the daemon's call order, up to the writer lease."""
    eager = apply_all(learning_db, memory_db)
    _engine_schema_bootstrap(memory_db)
    apply_all(learning_db, memory_db)
    deferred = apply_deferred(learning_db, memory_db)
    return eager, deferred


def _generations(snapshots: Path) -> list[str]:
    if not snapshots.exists():
        return []
    stamps = {
        m.group(1)
        for p in snapshots.iterdir()
        if (m := _STAMP.search(p.name))
    }
    return sorted(stamps)


def _remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _clear_snapshots(snapshots: Path) -> None:
    for child in list(snapshots.iterdir()):
        child.unlink()


def _forget_migration(db: Path, name: str) -> None:
    """Make one recorded migration pending again, as a new release would."""
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("DELETE FROM migration_log WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def store(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A fully migrated store with no snapshots, like a running 4.1.17 install."""
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _start(learning_db, memory_db)
    _start(learning_db, memory_db)
    snapshots = tmp_path / "pre-migration-snapshots"
    _clear_snapshots(snapshots)
    return learning_db, memory_db, snapshots


class TestOneStartWritesOneCopy:
    def test_a_start_with_learning_db_missing_writes_one_copy(self, store) -> None:
        """The exact shape observed: memory.db present, learning.db gone."""
        learning_db, memory_db, snapshots = store
        for suffix in ("", "-wal", "-shm"):
            _remove_file(Path(f"{learning_db}{suffix}"))

        eager, deferred = _start(learning_db, memory_db)

        assert "M027_transferable_patterns_profile" in deferred["applied"], (
            "fixture is wrong: the deferred pass had nothing to protect"
        )
        assert eager["details"].get("_backup") == str(snapshots)
        assert deferred["details"].get("_deferred_backup") == str(snapshots)
        gens = _generations(snapshots)
        assert len(gens) == 1, (
            f"one start wrote {len(gens)} copies of the whole store: {gens}"
        )

    def test_an_upgrade_with_work_in_both_passes_writes_one_copy(self, store) -> None:
        learning_db, memory_db, snapshots = store
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")

        eager, deferred = _start(learning_db, memory_db)

        assert "M004_cross_platform_sync_log" in eager["applied"]
        assert "M030_entity_explorer_indexes" in deferred["applied"]
        assert len(_generations(snapshots)) == 1

    def test_the_engine_initialisation_reuses_the_daemons_copy(
        self, store, monkeypatch,
    ) -> None:
        """The real engine path, not a stand-in: daemon pass, then initialize()."""
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.core.engine_capabilities import Capabilities
        from superlocalmemory.storage.models import Mode

        learning_db, memory_db, snapshots = store
        for suffix in ("", "-wal", "-shm"):
            _remove_file(Path(f"{learning_db}{suffix}"))
        monkeypatch.setattr(MemoryEngine, "_try_init_proxy", lambda self: None)
        config = SLMConfig.for_mode(Mode.A, base_dir=memory_db.parent)

        # unified_daemon.py resolves both paths before its eager pass.
        apply_all(learning_db.resolve(), memory_db.resolve())
        engine = MemoryEngine(config, capabilities=Capabilities.LIGHT)
        try:
            engine.initialize()
        finally:
            engine.close()

        assert len(_generations(snapshots)) == 1

    def test_one_store_reached_by_two_spellings_is_one_store(
        self, store, tmp_path: Path,
    ) -> None:
        """The daemon resolves its paths; the engine joins them from config.

        On macOS a temporary or relocated data folder is often reached through a
        symlink, so the two passes can name one store two ways.
        """
        learning_db, memory_db, snapshots = store
        alias = tmp_path.parent / f"{tmp_path.name}-alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")

        apply_all(alias / learning_db.name, alias / memory_db.name)
        deferred = apply_deferred(learning_db, memory_db)

        assert "M030_entity_explorer_indexes" in deferred["applied"]
        assert len(_generations(snapshots)) == 1

    def test_the_previous_upgrades_copy_survives_the_next_start(self, store) -> None:
        """Retention keeps two generations: this start's and the one before it."""
        learning_db, memory_db, snapshots = store
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")
        _start(learning_db, memory_db)
        # The earliest copy of that start is the one taken before it changed
        # anything -- the previous upgrade's rollback point.
        rollback_point = _generations(snapshots)[0]

        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")
        _start(learning_db, memory_db)

        gens = _generations(snapshots)
        assert rollback_point in gens, (
            "the copy taken before the previous upgrade was evicted by the "
            f"next start: kept {gens}"
        )
        assert len(gens) == 2


class TestWhenTheCopyMustNotBeReused:
    def test_a_deferred_pass_after_ordinary_writes_copies_for_itself(
        self, store,
    ) -> None:
        """The daemon's own deferred pass runs after the writer lease and replay."""
        learning_db, memory_db, snapshots = store
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _start(learning_db, memory_db)
        assert len(_generations(snapshots)) == 1

        conn = sqlite3.connect(str(memory_db))
        try:
            conn.execute(
                "INSERT INTO memories (memory_id, profile_id, content) "
                "VALUES ('after-start', 'default', 'written after the start')"
            )
            conn.commit()
        finally:
            conn.close()
        _forget_migration(memory_db, "M030_entity_explorer_indexes")

        later = apply_deferred(learning_db, memory_db)

        assert "M030_entity_explorer_indexes" in later["applied"]
        assert len(_generations(snapshots)) == 2, (
            "a deferred pass that ran after ordinary writes reused a copy that "
            "does not contain them"
        )

    def test_another_live_daemon_means_each_pass_copies(
        self, store, monkeypatch,
    ) -> None:
        """An old daemon still running can write between the two passes."""
        learning_db, memory_db, snapshots = store
        monkeypatch.setattr(migration_runner, "_foreign_live_daemon", lambda _db: 4242)
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")

        _start(learning_db, memory_db)

        assert len(_generations(snapshots)) == 2

    def test_a_live_writer_lease_held_elsewhere_means_each_pass_copies(
        self, store,
    ) -> None:
        """daemon.pid cannot show an old daemon: the new one overwrites it first.

        The new daemon publishes its own pid before its lifespan runs, so an old
        daemon still serving on another port is invisible there. The writer
        lease beside the store names whoever holds it.
        """
        import json
        import os

        learning_db, memory_db, snapshots = store
        lease = memory_db.with_name(memory_db.name + ".writer.lock")
        lease.write_text(json.dumps({
            "owner_id": "old-daemon", "pid": os.getppid(),
            "database": str(memory_db), "claimed_at_ms": 1,
        }), encoding="utf-8")
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")

        _start(learning_db, memory_db)

        assert len(_generations(snapshots)) == 2

    def test_a_lease_left_by_a_dead_daemon_does_not_block_reuse(self, store) -> None:
        import json

        learning_db, memory_db, snapshots = store
        lease = memory_db.with_name(memory_db.name + ".writer.lock")
        lease.write_text(json.dumps({"pid": 2 ** 22 + 12345, "claimed_at_ms": 1}), encoding="utf-8")
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")

        _start(learning_db, memory_db)

        assert len(_generations(snapshots)) == 1

    def test_a_copy_that_disappeared_is_not_counted_on(self, store) -> None:
        learning_db, memory_db, snapshots = store
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")
        apply_all(learning_db, memory_db)
        _clear_snapshots(snapshots)

        deferred = apply_deferred(learning_db, memory_db)

        assert "M030_entity_explorer_indexes" in deferred["applied"]
        assert len(_generations(snapshots)) == 1, (
            "the deferred pass changed the store with no copy on disk"
        )

    def test_a_dry_run_does_not_use_up_the_copy(self, store) -> None:
        learning_db, memory_db, snapshots = store
        _forget_migration(memory_db, "M004_cross_platform_sync_log")
        _forget_migration(memory_db, "M030_entity_explorer_indexes")
        apply_all(learning_db, memory_db)
        apply_deferred(learning_db, memory_db, dry_run=True)

        deferred = apply_deferred(learning_db, memory_db)

        assert "M030_entity_explorer_indexes" in deferred["applied"]
        assert len(_generations(snapshots)) == 1
