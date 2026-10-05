# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every safety copy taken before an update is described, checksummed and proven.

A copy that cannot be shown to hold the store must stop the update before it
changes anything: that copy is the only way back for a user who cannot read a
log or run a Python snippet.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from contextlib import closing
from pathlib import Path

import pytest

from superlocalmemory.storage import _snapshot_manifest as sm
from superlocalmemory.storage import backup
from superlocalmemory.storage import migration_runner as mr

from ._upgrade_store import add_memory, as_4118, columns, current_store, fact_ids


def _simple_store(root: Path, facts: int = 3) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    memory_db, learning_db = root / "memory.db", root / "learning.db"
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE memories (memory_id TEXT PRIMARY KEY, content TEXT)")
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT)")
        conn.executemany("INSERT INTO atomic_facts VALUES (?, ?)",
                         [(f"f{i}", f"fact {i}") for i in range(facts)])
        conn.execute("INSERT INTO memories VALUES ('m1', 'hello')")
        conn.commit()
    with closing(sqlite3.connect(learning_db)) as conn:
        conn.execute("CREATE TABLE learning_signals (id INTEGER PRIMARY KEY)")
        conn.commit()
    return learning_db, memory_db


def _manifests(root: Path) -> list[Path]:
    return sorted(root.glob("manifest-*-pre-migration.json"))


def test_manifest_written_after_rename_with_sha256(tmp_path, monkeypatch) -> None:
    learning_db, memory_db = _simple_store(tmp_path)
    (tmp_path / ".last_version").write_text("4.1.18", encoding="utf-8")
    snaps = tmp_path / "pre-migration-snapshots"
    seen: list[tuple[bool, bool]] = []
    real = sm.write_generation_manifest

    def spy(backups_root, stamp, pairs, **kwargs):
        finals = all(Path(s).is_file() for s, _db in pairs)
        partials = any(Path(f"{s}.partial").exists() for s, _db in pairs)
        seen.append((finals, partials))
        return real(backups_root, stamp, pairs, **kwargs)

    monkeypatch.setattr(sm, "write_generation_manifest", spy)
    backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)

    assert seen == [(True, False)], "manifest must be written after both renames"
    [manifest_file] = _manifests(snaps)
    assert stat.S_IMODE(manifest_file.stat().st_mode) == 0o600
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["reason"] == "migration"
    assert manifest["from_version"] == "4.1.18"
    assert manifest["counts"]["facts"] == 3 and manifest["counts"]["memories"] == 1
    assert {f["db"] for f in manifest["files"]} == {"memory.db", "learning.db"}
    for entry in manifest["files"]:
        on_disk = hashlib.sha256((snaps / entry["snapshot"]).read_bytes()).hexdigest()
        assert entry["sha256"] == on_disk
        assert entry["size"] == (snaps / entry["snapshot"]).stat().st_size


def test_gc_prunes_manifest_with_its_generation(tmp_path) -> None:
    learning_db, memory_db = _simple_store(tmp_path)
    snaps = tmp_path / "pre-migration-snapshots"
    for _ in range(3):
        backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)
    assert len(_manifests(snaps)) == 3
    oldest = _manifests(snaps)[0]

    backup._gc_old_backups(snaps, keep=2)

    left = _manifests(snaps)
    assert len(left) == 2 and oldest not in left
    for manifest_file in left:
        for entry in json.loads(manifest_file.read_text(encoding="utf-8"))["files"]:
            assert (snaps / entry["snapshot"]).is_file()


def test_partial_cleanup_touches_only_matching_files(tmp_path) -> None:
    root = tmp_path / "pre-migration-snapshots"
    root.mkdir()
    old = time.time() - 7200
    stale = "memory-20260101-000000-000001-pre-migration.db.partial"
    names = {
        stale: old, stale + "-wal": old, stale + "-shm": old,
        "learning-20260101-000000-000009-pre-migration.db.partial": time.time(),  # in progress
        "notes.partial": old,
        "random.db.partial": old,
        "memory-20260101-000000-000002-pre-migration.db": old,
    }
    for name, mtime in names.items():
        (root / name).write_bytes(b"x")
        os.utime(root / name, (mtime, mtime))

    removed = backup._cleanup_stale_partials(root)

    assert sorted(p.name for p in removed) == sorted([stale, stale + "-wal", stale + "-shm"])
    assert sorted(p.name for p in root.iterdir()) == sorted(
        set(names) - {stale, stale + "-wal", stale + "-shm"})


def test_a_new_copy_first_clears_an_interrupted_one(tmp_path) -> None:
    learning_db, memory_db = _simple_store(tmp_path)
    snaps = tmp_path / "pre-migration-snapshots"
    snaps.mkdir()
    stale = snaps / "memory-20260101-000000-000001-pre-migration.db.partial"
    stale.write_bytes(b"half a copy")
    os.utime(stale, (time.time() - 7200,) * 2)

    backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)

    assert not stale.exists()


def test_verify_snapshot_detects_a_flipped_byte(tmp_path) -> None:
    learning_db, memory_db = _simple_store(tmp_path, facts=200)
    snaps = tmp_path / "pre-migration-snapshots"
    backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)
    entry = next(f for f in json.loads(_manifests(snaps)[0].read_text(encoding="utf-8"))["files"]
                 if f["db"] == "memory.db")
    snapshot = snaps / entry["snapshot"]
    backup.verify_snapshot(snapshot, entry["sha256"])          # intact: passes

    data = bytearray(snapshot.read_bytes())
    data[len(data) // 2] ^= 0xFF
    snapshot.write_bytes(bytes(data))

    with pytest.raises(backup.SnapshotUnusableError, match="changed after it was taken"):
        backup.verify_snapshot(snapshot, entry["sha256"])


def test_the_update_does_not_run_on_a_copy_that_holds_nothing(tmp_path, monkeypatch) -> None:
    """A copy that passes SQLite's own check but holds none of the store."""
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["The release key lives in the vault", "Ship on Fridays"])
    as_4118(learning_db, memory_db)
    before = fact_ids(memory_db)

    def empty_copy(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(dest)) as conn:
            conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
            conn.commit()

    monkeypatch.setattr(backup, "_backup_via_sqlite_api", empty_copy)
    with pytest.raises(backup.SnapshotUnusableError, match="holds 0 memories"):
        mr.apply_all(learning_db, memory_db)

    assert "memory_kind" not in columns(memory_db), "M052 ran without a usable copy"
    assert fact_ids(memory_db) == before


def test_a_store_with_unflushed_log_is_copied_whole(tmp_path) -> None:
    """Committed rows still sitting in the -wal are in the copy (CRIT: WAL-hot store)."""
    learning_db, memory_db = _simple_store(tmp_path, facts=5)
    writer = sqlite3.connect(memory_db)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.executemany("INSERT INTO atomic_facts VALUES (?, ?)",
                           [(f"w{i}", "only in the log") for i in range(50)])
        writer.commit()
        assert Path(f"{memory_db}-wal").stat().st_size > 0
        snaps = tmp_path / "pre-migration-snapshots"
        backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)
    finally:
        writer.close()

    manifest = json.loads(_manifests(snaps)[0].read_text(encoding="utf-8"))
    assert manifest["counts"]["facts"] == 55
    entry = next(f for f in manifest["files"] if f["db"] == "memory.db")
    backup.verify_snapshot(snaps / entry["snapshot"], entry["sha256"])
