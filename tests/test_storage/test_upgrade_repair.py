# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The one-time repair runs once, after the update, and only with a verified copy to go back to."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from superlocalmemory.storage import backup
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import as_4118, clear_snapshots, current_store

ORIGINAL = "Backups go to Agentic_official/slm-backups/pre-4.1.18-x with sha256."
BLANKED = "Backups go to [REDACTED:ENTROPY:.18-]x with sha256."
MISMATCH = "queryable ingestion memory content mismatch"


def _content(memory_db: Path) -> str:
    with closing(sqlite3.connect(memory_db)) as conn:
        return conn.execute("SELECT content FROM memories WHERE memory_id='m1'").fetchone()[0]


def _damaged_4118_store(root: Path) -> tuple[Path, Path]:
    learning_db, memory_db = current_store(root)
    as_4118(learning_db, memory_db)
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                     "VALUES ('m1', 'default', ?)", (BLANKED,))
        conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                     "VALUES ('f1', 'm1', 'default', ?)", (BLANKED,))
        conn.execute(
            "INSERT INTO ingestion_operations (operation_id, profile_id, source_type, "
            "idempotency_key, source_hash, raw_content, state, last_error, "
            "queryable_fact_ids_json) VALUES ('op1', 'default', 'http', 'k1', 'h', ?, "
            "'failed', ?, ?)", (ORIGINAL, MISMATCH, json.dumps(["f1"])))
        conn.commit()
    return learning_db, memory_db


@pytest.fixture()
def upgraded(tmp_path):
    """A damaged 4.1.18 store after the 4.1.19 update ran (copy taken, M052 applied)."""
    learning_db, memory_db = _damaged_4118_store(tmp_path)
    assert "M052_memory_kinds" in mr.apply_all(learning_db, memory_db)["applied"]
    return tmp_path, learning_db, memory_db


def test_repair_does_not_run_without_a_verified_snapshot(upgraded) -> None:
    root, _learning_db, memory_db = upgraded
    clear_snapshots(root)
    result = ur.run_store_repair_once(root, memory_db)
    assert result["status"] == "waiting_for_snapshot"
    assert _content(memory_db) == BLANKED
    assert not (root / "upgrade-repair.json").exists()


def test_repair_does_not_run_on_a_copy_that_changed(upgraded) -> None:
    root, _learning_db, memory_db = upgraded
    [point] = ur.list_restore_points(root)
    data = bytearray(point.memory_snapshot.read_bytes())
    data[len(data) // 2] ^= 0xFF
    point.memory_snapshot.write_bytes(bytes(data))

    assert ur.run_store_repair_once(root, memory_db)["status"] == "waiting_for_snapshot"
    assert _content(memory_db) == BLANKED


def test_repair_waits_for_the_update(tmp_path) -> None:
    learning_db, memory_db = _damaged_4118_store(tmp_path)
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    assert ur.run_store_repair_once(tmp_path, memory_db)["status"] == "waiting_for_update"
    assert _content(memory_db) == BLANKED


def test_repair_runs_once_after_snapshot_and_m052(upgraded) -> None:
    root, learning_db, memory_db = upgraded
    [point] = ur.list_restore_points(root)

    first = ur.run_store_repair_once(root, memory_db)

    assert first["status"] == "ran" and first["point_id"] == point.point_id
    assert first["result"]["memories_restored"] == 1
    assert _content(memory_db) == ORIGINAL
    preview = ur.preview_restore(point.point_id, data_root=root, memory_db=memory_db)
    assert preview.repair["memories_restored"] == 1, "the preview says what a restore undoes"
    assert ur.list_restore_points(root)[0].repair["memories_restored"] == 1

    second = ur.run_store_repair_once(root, memory_db)            # the next start
    assert second["status"] == "already_done"
    assert _content(memory_db) == ORIGINAL

    # A user who restores to before the update is not repaired again behind their back.
    ur.request_restore(point.point_id, requested_by="t", data_root=root, memory_db=memory_db)
    assert ur.perform_pending_restore(root, memory_db, learning_db).status == "restored"
    mr.apply_all(learning_db, memory_db)
    assert ur.run_store_repair_once(root, memory_db)["status"] == "already_done"
    assert _content(memory_db) == BLANKED
