# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A fresh install's empty install-time copy (audit 4.1.20 L5).

A fresh install copies its database before the first migration, when it holds
no memory tables at all. That copy can never be restored, yet it was listed as
a restore point, and the one-time repair waited on it forever, logging "does
not verify" on every start.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import add_memory, as_4118, current_store


def _install_time_copy(root: Path) -> None:
    """The copy a fresh install takes: a database holding only memory_events."""
    blank = root / "blank"
    blank.mkdir()
    memory_db, learning_db = blank / "memory.db", blank / "learning.db"
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("CREATE TABLE memory_events (id INTEGER PRIMARY KEY)")
        conn.commit()
    sqlite3.connect(learning_db).close()
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=root / "pre-migration-snapshots")


def _fresh_install(root: Path) -> Path:
    _learning_db, memory_db = current_store(root)
    _install_time_copy(root)
    time.sleep(1.1)          # row timestamps have one-second resolution
    add_memory(memory_db, "m1", ["The first thing this install ever saved"])
    return memory_db


def test_an_empty_install_time_copy_is_not_a_restore_point(tmp_path) -> None:
    _fresh_install(tmp_path)
    assert list((tmp_path / "pre-migration-snapshots").glob("memory-*.db")), "copy exists"
    assert ur.list_restore_points(tmp_path) == []


def test_a_copy_with_memories_is_still_listed(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["kept"])
    _install_time_copy(tmp_path)
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    assert point.facts == 1


def test_fresh_install_repair_is_settled_once_without_a_warning(tmp_path, caplog) -> None:
    memory_db = _fresh_install(tmp_path)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        first = ur.run_store_repair_once(tmp_path, memory_db)
    assert first["status"] == "nothing_to_repair"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "does not verify" not in caplog.text
    assert ur.run_store_repair_once(tmp_path, memory_db)["status"] == "already_done"


def test_a_store_with_rows_older_than_the_update_still_waits(tmp_path) -> None:
    """Rows that predate the update may carry old-build damage: no shortcut."""
    from superlocalmemory.storage import migration_runner as mr

    learning_db, memory_db = current_store(tmp_path)
    as_4118(learning_db, memory_db)
    add_memory(memory_db, "m1", ["saved under 4.1.18"])
    time.sleep(1.1)
    mr.apply_all(learning_db, memory_db)
    for child in sorted((tmp_path / "pre-migration-snapshots").iterdir()):
        child.unlink()
    assert ur.run_store_repair_once(tmp_path, memory_db)["status"] == "waiting_for_snapshot"
