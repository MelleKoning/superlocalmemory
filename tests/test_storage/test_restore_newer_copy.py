# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A copy made by a newer build is never restored by this one (L1-04).

Path: a newer version (store version 53) prepares a downgrade, which takes a
copy stamped 53; the person installs this version and sees that copy listed.
Restoring it would leave a store this build refuses to open at its next start.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._durable_json import write_json_atomic
from superlocalmemory.storage._schema_version import SUPPORTED_SCHEMA_VERSION, read_schema_version

from ._upgrade_store import add_memory, current_store, file_digest

NEWER = SUPPORTED_SCHEMA_VERSION + 1


def _stamp(db: Path, version: int) -> None:
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("UPDATE slm_schema_version SET version=? WHERE id=1", (version,))
        conn.commit()


def _newer_copy(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"])
    _stamp(learning_db, NEWER)
    _stamp(memory_db, NEWER)
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots",
                                 reason="pre-downgrade")
    _stamp(learning_db, SUPPORTED_SCHEMA_VERSION)
    _stamp(memory_db, SUPPORTED_SCHEMA_VERSION)
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def test_listed_as_not_restorable_with_the_reason(tmp_path) -> None:
    _l, _m, point = _newer_copy(tmp_path)
    assert point.schema_version == NEWER and point.restorable is False
    assert "newer version" in point.not_restorable_reason
    assert point.as_dict()["restorable"] is False


def test_preview_and_request_refuse(tmp_path) -> None:
    _l, memory_db, point = _newer_copy(tmp_path)
    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)
    assert not preview.restorable and any("newer version" in p for p in preview.problems)

    with pytest.raises(ur.RestoreRefusedError, match="newer version"):
        ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                           memory_db=memory_db)
    assert not (tmp_path / "restore-intent.json").exists()


def test_boot_refuses_a_request_for_it_and_changes_nothing(tmp_path) -> None:
    learning_db, memory_db, point = _newer_copy(tmp_path)
    write_json_atomic(tmp_path / "restore-intent.json", {
        "format": 1, "point_id": point.point_id, "requested_by": "t", "reimport": True,
        "delta_dir": "", "stage": "requested"})
    before = file_digest(memory_db)

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "refused" and "newer version" in outcome.message
    assert file_digest(memory_db) == before
    assert read_schema_version(memory_db) == SUPPORTED_SCHEMA_VERSION
    assert not (tmp_path / "restore-intent.json").exists(), "set aside, not retried forever"


def test_an_older_or_current_copy_stays_restorable(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    assert point.restorable and point.schema_version == SUPPORTED_SCHEMA_VERSION
