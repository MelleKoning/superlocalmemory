# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""BackupCoordinator restore against WAL-mode stores with un-checkpointed rows.

Every live store here is in WAL mode, and a connection commits rows with
auto-checkpoint off and stays open, so those committed rows exist only in the
store's ``-wal``. That is the normal state of a store the daemon is using.

Two defects this guards against, both reproduced before the fix:
  * the pre-restore safety copy was a file copy of ``memory.db``, which omits
    every committed frame still in ``memory.db-wal``;
  * the restore renamed a copy over ``memory.db`` and left the old ``-wal``
    beside it. SQLite then read the old store's frames over the restored
    pages: the restore reported success and changed nothing, or, when the live
    store had moved on since the backup, left "database disk image is
    malformed".

Real temp directories and real SQLite databases only. Never the user's data
directory.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from superlocalmemory.infra.backup import BackupCoordinator, BackupRestoreError

_PAYLOAD = "x" * 300
_SUBSET: tuple[str, ...] = ("memory.db", "learning.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wal_store(path: Path, tag: str, rows: int, *, replace: bool = False) -> None:
    """Commit ``rows`` rows tagged ``tag`` into a WAL store, then close it.

    Closing the last connection checkpoints, so these rows are in the main file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, tag TEXT, body TEXT)")
        if replace:
            conn.execute("DELETE FROM t")
        conn.executemany("INSERT INTO t (tag, body) VALUES (?, ?)", [(tag, _PAYLOAD)] * rows)
        conn.commit()
    finally:
        conn.close()


def _hold_uncheckpointed(path: Path, tag: str, rows: int) -> sqlite3.Connection:
    """Commit rows that exist only in ``path``'s -wal, and keep the connection open.

    The caller must close the returned connection.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.executemany("INSERT INTO t (tag, body) VALUES (?, ?)", [(tag, _PAYLOAD)] * rows)
    conn.commit()
    return conn


def _tags(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(conn.execute("SELECT tag, count(*) FROM t GROUP BY tag ORDER BY tag"))


def _live_tags(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(path))
    try:
        return _tags(conn)
    finally:
        conn.close()


def _tags_of_file_alone(path: Path, scratch: Path) -> dict[str, int]:
    """Tags readable from ``path``'s bytes alone, without any -wal beside it."""
    scratch.mkdir(parents=True, exist_ok=True)
    alone = scratch / f"{path.name}.alone"
    shutil.copyfile(path, alone)
    conn = sqlite3.connect(f"{alone.absolute().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        return _tags(conn)
    finally:
        conn.close()


def _dump(path: Path, *, immutable: bool = False) -> list[str]:
    """Logical content of a database: its schema and every row, as SQL."""
    if immutable:
        conn = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro&immutable=1", uri=True)
    else:
        conn = sqlite3.connect(str(path))
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def _integrity(path: Path) -> str:
    """``PRAGMA integrity_check`` result; a database too damaged to open says why."""
    conn = sqlite3.connect(str(path))
    try:
        return "; ".join(r[0] for r in conn.execute("PRAGMA integrity_check"))
    except sqlite3.DatabaseError as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


def _wal_bytes(path: Path) -> int:
    wal = path.with_name(path.name + "-wal")
    return wal.stat().st_size if wal.exists() else 0


def _fail_live_writes(
    coord: BackupCoordinator, monkeypatch: pytest.MonkeyPatch, failing_calls: set[int]
) -> list[str]:
    """Make the listed (1-based) writes into live stores raise; the rest are real.

    Writes are counted across Phase C and the rollback. Returns target names in
    call order.
    """
    real_write = coord._write_into_live_store
    calls: list[str] = []

    def _write(source: Path, target: Path) -> None:
        calls.append(target.name)
        if len(calls) in failing_calls:
            raise OSError(f"simulated disk error writing {target.name}")
        real_write(source, target)

    monkeypatch.setattr(coord, "_write_into_live_store", _write)
    return calls


@pytest.fixture()
def env(tmp_path: Path) -> tuple[Path, Path, BackupCoordinator]:
    """Two WAL live stores and a coordinator; memory.db holds 400 'A' rows."""
    base_dir = tmp_path / "live"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True)
    _wal_store(base_dir / "memory.db", "A", 400)
    _wal_store(base_dir / "learning.db", "L", 50)
    coord = BackupCoordinator(
        managed_databases=_SUBSET, base_dir=base_dir, backup_dir=backup_dir,
    )
    return base_dir, backup_dir, coord


# ---------------------------------------------------------------------------
# Phase B: the pre-restore safety copy
# ---------------------------------------------------------------------------

def test_pre_restore_snapshot_contains_uncheckpointed_rows(
    env: tuple[Path, Path, BackupCoordinator], tmp_path: Path
) -> None:
    base_dir, _backup_dir, coord = env
    live = base_dir / "memory.db"
    manifest = coord.create_backup_set()

    holder = _hold_uncheckpointed(live, "B", 300)
    try:
        # Guard: the B rows really are committed and really are only in the
        # -wal. Without this the test could pass against a raw file copy.
        assert _wal_bytes(live) > 0
        assert _tags(holder) == {"A": 400, "B": 300}
        assert _tags_of_file_alone(live, tmp_path / "scratch") == {"A": 400}

        # The snapshot is removed after a successful restore, so read it just
        # before cleanup — from its own bytes, as a rollback would use it.
        snapshot_tags: dict[str, dict[str, int]] = {}
        real_cleanup = coord._cleanup_pre_restore_snapshots

        def _capture_then_cleanup(snapshot_map, lance_snapshot):
            for target, snapshot in snapshot_map.items():
                if snapshot.exists():
                    snapshot_tags[target.name] = _tags_of_file_alone(
                        snapshot, tmp_path / "scratch")
            return real_cleanup(snapshot_map, lance_snapshot)

        coord._cleanup_pre_restore_snapshots = _capture_then_cleanup
        coord.restore_from_manifest(manifest)
    finally:
        holder.close()

    assert snapshot_tags["memory.db"] == {"A": 400, "B": 300}, (
        "pre-restore snapshot is missing rows that were committed but still "
        "in memory.db-wal"
    )
    assert not list(base_dir.glob("*.pre_restore*")), "snapshot residue left behind"


# ---------------------------------------------------------------------------
# Phase C: the restore write
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("live_moved_on", [False, True], ids=["same-base", "live-moved-on"])
def test_restore_over_hot_wal_matches_backup_and_is_intact(
    env: tuple[Path, Path, BackupCoordinator], tmp_path: Path, live_moved_on: bool
) -> None:
    base_dir, _backup_dir, coord = env
    live = base_dir / "memory.db"
    manifest = coord.create_backup_set()
    backup_file = Path(next(e.file_path for e in manifest.stores if e.store_name == "memory.db"))
    backup_dump = _dump(backup_file, immutable=True)

    if live_moved_on:
        # The live store changed shape after the backup and was checkpointed,
        # so the frames about to sit in the -wal describe a different tree.
        _wal_store(live, "A2", 2000, replace=True)
    holder = _hold_uncheckpointed(live, "B", 300)
    try:
        assert _wal_bytes(live) > 0
        assert _tags(holder)["B"] == 300

        coord.restore_from_manifest(manifest)

        assert _integrity(live) == "ok"
        assert _dump(live) == backup_dump, "live store does not hold the backup's content"
        # A reader that was open throughout sees the restored content too.
        assert _tags(holder) == {"A": 400}
    finally:
        holder.close()

    # Once the last connection closes, nothing stale is checkpointed back in.
    assert _integrity(live) == "ok"
    assert _dump(live) == backup_dump
    # The backup set was read, never written: no companions beside it.
    assert not [p.name for p in backup_file.parent.iterdir()
                if p.name.endswith(("-wal", "-shm"))]


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def test_rollback_restores_uncheckpointed_rows(
    env: tuple[Path, Path, BackupCoordinator], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir, _backup_dir, coord = env
    live = base_dir / "memory.db"
    manifest = coord.create_backup_set()
    learning_before = _dump(base_dir / "learning.db")

    holder = _hold_uncheckpointed(live, "B", 300)
    try:
        before = _dump(live)
        assert _wal_bytes(live) > 0

        # memory.db is restored, then learning.db fails: memory.db must be
        # rolled back to what was live, including the rows only in its -wal.
        calls = _fail_live_writes(coord, monkeypatch, failing_calls={2})
        with pytest.raises(BackupRestoreError, match="rolled back"):
            coord.restore_from_manifest(manifest)

        assert calls == ["memory.db", "learning.db", "memory.db"]
        assert _integrity(live) == "ok"
        assert _dump(live) == before
        assert _tags(holder) == {"A": 400, "B": 300}
    finally:
        holder.close()

    assert _dump(live) == before
    assert _dump(base_dir / "learning.db") == learning_before
    assert not list(base_dir.glob("*.pre_restore*"))


def test_failed_rollback_keeps_snapshots(
    env: tuple[Path, Path, BackupCoordinator], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir, _backup_dir, coord = env
    manifest = coord.create_backup_set()
    _wal_store(base_dir / "memory.db", "C", 10)
    live_before = _dump(base_dir / "memory.db")

    # learning.db fails, then rolling memory.db back fails too.
    _fail_live_writes(coord, monkeypatch, failing_calls={2, 3})
    with pytest.raises(BackupRestoreError, match="rollback encountered errors") as info:
        coord.restore_from_manifest(manifest)

    snapshot = base_dir / "memory.db.pre_restore"
    assert snapshot.exists(), "the only copy of the pre-restore state was deleted"
    assert str(snapshot) in str(info.value)
    assert _dump(snapshot, immutable=True) == live_before


# ---------------------------------------------------------------------------
# Failure modes the SQLite write introduces
# ---------------------------------------------------------------------------

def test_restore_gives_up_when_a_writer_holds_the_store(
    env: tuple[Path, Path, BackupCoordinator]
) -> None:
    base_dir, backup_dir, _coord = env
    coord = BackupCoordinator(
        managed_databases=_SUBSET, base_dir=base_dir, backup_dir=backup_dir,
        lock_wait_seconds=0.5,
    )
    live = base_dir / "memory.db"
    manifest = coord.create_backup_set()
    _wal_store(live, "C", 10)
    before = _dump(live)

    writer = sqlite3.connect(str(live))
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(BackupRestoreError, match="stayed locked"):
            coord.restore_from_manifest(manifest)
        assert time.monotonic() - started < 10, "restore waited far past its lock bound"
    finally:
        writer.rollback()
        writer.close()

    assert _dump(live) == before
    assert _integrity(live) == "ok"
    assert not list(base_dir.glob("*.pre_restore*"))


def test_page_size_mismatch_into_wal_store_is_refused_cleanly(
    env: tuple[Path, Path, BackupCoordinator]
) -> None:
    base_dir, _backup_dir, coord = env
    live = base_dir / "memory.db"
    manifest = coord.create_backup_set()

    # Rebuild the live store with a different page size, back in WAL mode.
    conn = sqlite3.connect(str(live))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA page_size=8192")
        conn.execute("VACUUM")
        conn.execute("PRAGMA journal_mode=WAL")
        assert conn.execute("PRAGMA page_size").fetchone()[0] == 8192
    finally:
        conn.close()
    _wal_store(live, "C", 10)
    before = _dump(live)

    with pytest.raises(BackupRestoreError, match="page size"):
        coord.restore_from_manifest(manifest)

    assert _dump(live) == before
    assert _integrity(live) == "ok"
