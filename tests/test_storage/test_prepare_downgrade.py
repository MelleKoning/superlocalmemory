# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Prepare to go back to 4.1.18: only when safe, copied first, undone by any other build."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import superlocalmemory.storage._schema_version as sv
from superlocalmemory.storage import _snapshot_manifest as sm
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._restore_types import DowngradeRefusedError
from superlocalmemory.storage.migrations import M052_memory_kinds as M052

from ._upgrade_store import current_store

#: The schema this build stamps (54 since M054, saved views).
_CURRENT = sv.SUPPORTED_SCHEMA_VERSION


@pytest.fixture()
def store(tmp_path, monkeypatch):
    learning_db, memory_db = current_store(tmp_path)
    (tmp_path / ".last_version").write_text("4.1.19", encoding="utf-8")
    monkeypatch.setattr(sm, "package_version", lambda: "4.1.19")
    monkeypatch.setattr(sv, "_detect_all_installs", lambda: [])
    assert sv.read_schema_version(memory_db) == _CURRENT
    return tmp_path, learning_db, memory_db


def _prepare(root: Path, learning_db: Path, memory_db: Path):
    return ur.prepare_downgrade(requested_by="test", data_root=root,
                                memory_db=memory_db, learning_db=learning_db)


def _next_start(learning_db: Path, memory_db: Path) -> dict:
    mr.apply_all(learning_db, memory_db)
    return mr.apply_deferred(learning_db, memory_db)


def test_refuses_after_a_breaking_migration(store, monkeypatch) -> None:
    root, learning_db, memory_db = store
    monkeypatch.setattr(M052, "BREAKING_VERSION", 52)
    with pytest.raises(DowngradeRefusedError, match="not safe"):
        _prepare(root, learning_db, memory_db)
    monkeypatch.setattr(M052, "BREAKING_VERSION", 0)
    monkeypatch.delattr(M052, "DOWNGRADE_FLOOR")        # a change that cannot be undone
    with pytest.raises(DowngradeRefusedError, match="cannot be undone"):
        _prepare(root, learning_db, memory_db)
    assert sv.read_schema_version(memory_db) == _CURRENT
    assert not (root / ".downgrade-prepared").exists()
    assert ur.list_restore_points(root) == [], "refused before copying anything"


def test_lowers_stamp_and_writes_marker(store) -> None:
    root, learning_db, memory_db = store
    report = _prepare(root, learning_db, memory_db)
    assert report.prepared and report.target_schema == 51
    assert sv.read_schema_version(memory_db) == 51 == sv.read_schema_version(learning_db)
    marker = json.loads((root / ".downgrade-prepared").read_text(encoding="utf-8"))
    assert marker["prepared_by"] == "4.1.19" and marker["target_schema"] == 51
    assert marker["last_version"] == "4.1.19"
    [point] = ur.list_restore_points(root)
    assert point.point_id == report.point_id and point.reason == "pre-downgrade"
    ur.verify_point(point)


def test_runner_holds_stamp_while_marker_is_valid(store) -> None:
    root, learning_db, memory_db = store
    _prepare(root, learning_db, memory_db)
    result = _next_start(learning_db, memory_db)
    assert result["failed"] == [], result["details"]
    assert "held" in result["details"]["schema_version_stamp"]
    assert sv.read_schema_version(memory_db) == 51
    assert (root / ".downgrade-prepared").exists()


def test_marker_dropped_after_another_build_ran(store) -> None:
    root, learning_db, memory_db = store
    _prepare(root, learning_db, memory_db)
    version = root / ".last_version"
    version.write_text("4.1.18", encoding="utf-8")                      # the older build started ...
    version.write_text("4.1.19", encoding="utf-8")                      # ... and this one again after it
    later = version.stat().st_mtime + 5
    os.utime(version, (later, later))

    _next_start(learning_db, memory_db)

    assert sv.read_schema_version(memory_db) == _CURRENT
    assert not (root / ".downgrade-prepared").exists()


def test_51_ceiling_guard_accepts_prepared_store(store, monkeypatch) -> None:
    root, learning_db, memory_db = store
    monkeypatch.setattr(sv, "SUPPORTED_SCHEMA_VERSION", 51)
    with pytest.raises(sv.SchemaVersionError):           # unprepared: an old build refuses
        sv.check_version_or_raise(memory_db)
    monkeypatch.setattr(sv, "SUPPORTED_SCHEMA_VERSION", _CURRENT)
    _prepare(root, learning_db, memory_db)
    monkeypatch.setattr(sv, "SUPPORTED_SCHEMA_VERSION", 51)
    sv.check_version_or_raise(memory_db)
    sv.check_version_or_raise(learning_db)


def test_cancel_restamps_on_next_start(store) -> None:
    root, learning_db, memory_db = store
    _prepare(root, learning_db, memory_db)
    assert ur.cancel_downgrade(root) is True
    result = _next_start(learning_db, memory_db)
    assert result["failed"] == []
    assert sv.read_schema_version(memory_db) == _CURRENT == sv.read_schema_version(learning_db)
    assert ur.cancel_downgrade(root) is False
