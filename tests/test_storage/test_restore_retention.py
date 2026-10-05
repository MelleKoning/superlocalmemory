# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Restore leftovers are bounded and covered by erasure (L1-12)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.infra.backup_obligations import BackupObligationStore
from superlocalmemory.storage import _restore_retention as rr
from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._durable_json import read_json

from ._upgrade_store import add_memory, add_profile, current_store


def _restored_with_one_new_memory(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"], fact_ids=["f1"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    add_memory(memory_db, "later", ["Written after the copy was taken"], fact_ids=["lf"])
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)
    assert outcome.status == "restored" and outcome.reimport_pending
    return learning_db, memory_db, Path(outcome.delta_dir)


def test_the_delta_goes_once_everything_is_back(tmp_path, monkeypatch) -> None:
    _l, memory_db, delta = _restored_with_one_new_memory(tmp_path)
    from superlocalmemory.storage import _restore_reimport as ri

    def readd(_engine, rows, _db, counts):
        counts["added"] += len(rows)
    monkeypatch.setattr(ri, "_reimport_memories", readd)

    report = ur.reimport_delta(SimpleNamespace(_db=SimpleNamespace(db_path=memory_db)),
                               delta, data_root=tmp_path)

    assert report.added == 1 and not delta.exists()
    outcome = read_json(tmp_path / "restore-outcome.json")
    assert outcome["reimport_pending"] is False and outcome["delta_removed"] is True
    assert not any((tmp_path / "restore-delta").iterdir())


@pytest.mark.parametrize("result", ["failed", "skipped_rejected", "skipped_unknown_profile"])
def test_a_delta_holding_a_memory_not_in_the_store_is_kept(tmp_path, monkeypatch,
                                                           result) -> None:
    _l, memory_db, delta = _restored_with_one_new_memory(tmp_path)
    from superlocalmemory.storage import _restore_reimport as ri

    def fail(_engine, rows, _db, counts):
        counts[result] += len(rows)
    monkeypatch.setattr(ri, "_reimport_memories", fail)

    ur.reimport_delta(SimpleNamespace(_db=SimpleNamespace(db_path=memory_db)), delta,
                      data_root=tmp_path)

    assert (delta / "memories.jsonl").exists(), "it holds a memory not in the store"
    assert rr.prune_restore_artifacts(tmp_path, now=time.time() + 400 * 86400) == []


def test_a_restore_with_nothing_to_add_back_leaves_no_delta(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"], fact_ids=["f1"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "restored" and not outcome.reimport_pending
    assert not any((tmp_path / "restore-delta").iterdir())


def _old_copy(safety: Path, stamp: str, days_old: float) -> Path:
    path = safety / f"memory-{stamp}-000000-before-restore.db"
    path.write_bytes(b"x")
    when = time.time() - days_old * 86400
    os.utime(path, (when, when))
    return path


def test_pre_restore_copies_are_pruned_but_the_newest_two_stay(tmp_path) -> None:
    safety = tmp_path / "pre-restore"
    safety.mkdir()
    copies = [_old_copy(safety, f"2026010{i}-120000", 60) for i in range(1, 5)]
    other = safety / "notes.txt"
    other.write_text("not ours", encoding="utf-8")

    removed = rr.prune_restore_artifacts(tmp_path)

    assert sorted(removed) == sorted(copies[:2])
    assert [p.exists() for p in copies] == [False, False, True, True] and other.exists()


def test_nothing_is_pruned_while_a_reimport_waits(tmp_path) -> None:
    _l, _m, delta = _restored_with_one_new_memory(tmp_path)
    safety = tmp_path / "pre-restore"
    copies = [_old_copy(safety, f"2025010{i}-120000", 60) for i in range(1, 5)]

    assert rr.prune_restore_artifacts(tmp_path) == []
    assert all(p.exists() for p in copies) and delta.is_dir()


def test_erasure_reaches_deltas_and_pre_restore_copies(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_profile(memory_db, "alice")
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"], fact_ids=["f1"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    add_memory(memory_db, "ma", ["Alice's address is 12 Elm St"], profile="alice",
               fact_ids=["fa1"])
    add_memory(memory_db, "mb", ["Standup moved to 10am"], fact_ids=["fb1"])
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)
    delta = Path(outcome.delta_dir)
    assert "Elm St" in (delta / "memories.jsonl").read_text(encoding="utf-8")
    counts: dict = {}

    GDPRCompliance._record_backup_obligations(None, data_root=tmp_path, profile_id="alice",
                                              erasure_id="e1", counts=counts)

    rows = [json.loads(line) for line in (delta / "memories.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["memory_id"] for r in rows] == ["mb"], "alice's words are gone from the delta"
    assert counts["restore_delta_rows_erased"] == 1
    assert counts["pre_restore_copies_registered"] == 1
    store = BackupObligationStore(tmp_path)
    [pending] = store.list_pending_for_profile("alice")
    assert Path(pending["snapshot_path"]).parent == tmp_path / "pre-restore"
