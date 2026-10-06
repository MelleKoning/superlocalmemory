# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Restore points, backups and remote secrets are readable by their owner only.

Releases before 4.1.20 left some of them ``0644`` (a 4.1.19 learning restore
point on a real install was). Every writer now makes a new copy owner-only
before writing into it, and each daemon start tightens what an older release
left behind, idempotently.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from superlocalmemory.infra import private_files
from superlocalmemory.infra.private_files import tighten_private_files

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")

REPO = Path(__file__).resolve().parents[2]


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _open_file(path: Path, mode: int = 0o644, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, mode)
    return path


def _old_install(root: Path) -> dict[str, Path]:
    """The files a pre-4.1.20 install could have left world-readable."""
    files = {
        "learning_restore_point": root / "pre-migration-snapshots"
        / "learning-20260901-101010-pre-migration.db",
        "memory_restore_point": root / "pre-migration-snapshots"
        / "memory-20260901-101010-pre-migration.db",
        "manifest": root / "pre-migration-snapshots" / "20260901-101010.manifest.json",
        "pre_restore": root / "pre-restore" / "memory-20260902-before-restore.db",
        "delta": root / "restore-delta" / "20260902" / "memories.jsonl",
        "backup": root / "backups" / "memory-20260903-010101.db",
        "backup_set": root / "backups" / "backup_abc" / "memory.db",
        "learning_backup": root / "learning.db.backup_20260903_010101",
        "remote_keys": root / "remote_keys.json",
        "remote_settings": root / "remote" / "remote.json",
        "tls_key": root / "remote" / "tls" / "server.key",
        "ca_key": root / "remote" / "tls" / "ca.key",
        "config": root / "config.json",
        "feedback_key": root / ".feedback-hash-key",
        "answer_check_key": root / "secrets" / "jev-typesafe.key",
    }
    for path in files.values():
        _open_file(path)
    for folder in ("pre-migration-snapshots", "pre-restore", "restore-delta",
                   "restore-delta/20260902", "backups", "backups/backup_abc", "secrets"):
        os.chmod(root / folder, 0o755)
    return files


def test_start_up_pass_makes_old_copies_and_secrets_owner_only(tmp_path) -> None:
    files = _old_install(tmp_path)
    unrelated = _open_file(tmp_path / "ui" / "index.html")
    certificate = _open_file(tmp_path / "remote" / "tls" / "server.pem")

    report = tighten_private_files(tmp_path)

    for name, path in files.items():
        assert _mode(path) == 0o600, f"{name} is still {oct(_mode(path))}"
    for folder in ("pre-migration-snapshots", "pre-restore", "restore-delta",
                   "restore-delta/20260902", "backups", "backups/backup_abc", "secrets"):
        assert _mode(tmp_path / folder) == 0o700, folder
    assert _mode(unrelated) == 0o644, "files that are not copies or secrets are left alone"
    assert _mode(certificate) == 0o644, "a public certificate is not a secret"
    assert not report.failed and not report.skipped
    assert str(files["learning_restore_point"]) in report.tightened


def test_start_up_pass_is_idempotent(tmp_path, monkeypatch) -> None:
    _old_install(tmp_path)
    tighten_private_files(tmp_path)
    calls: list[str] = []
    real_chmod = os.chmod
    monkeypatch.setattr(private_files.os, "chmod",
                        lambda p, m, **kw: (calls.append(str(p)), real_chmod(p, m, **kw)))
    report = tighten_private_files(tmp_path)
    assert calls == [] and not report, "an already private file was changed again"


def test_nothing_is_ever_widened(tmp_path) -> None:
    read_only = _open_file(tmp_path / "backups" / "memory-old.db", 0o400)
    group_read = _open_file(tmp_path / "backups" / "memory-g.db", 0o640)
    tighten_private_files(tmp_path)
    assert _mode(read_only) == 0o400
    assert _mode(group_read) == 0o600


def test_links_out_of_the_data_folder_are_not_followed(tmp_path) -> None:
    outside = _open_file(tmp_path / "elsewhere" / "someone-elses.db")
    root = tmp_path / "data"
    (root / "pre-restore").mkdir(parents=True)
    os.symlink(outside, root / "pre-restore" / "link.db")
    os.symlink(tmp_path / "elsewhere", root / "backups")
    os.chmod(tmp_path / "elsewhere", 0o755)
    tighten_private_files(root)
    assert _mode(outside) == 0o644, "a file reached through a link was changed"
    assert _mode(tmp_path / "elsewhere") == 0o755, "a folder reached through a link was changed"


def test_a_missing_data_folder_is_fine(tmp_path) -> None:
    assert not tighten_private_files(tmp_path / "nope")


def test_a_file_that_cannot_be_changed_is_reported_and_the_rest_continue(
        tmp_path, monkeypatch) -> None:
    files = _old_install(tmp_path)
    real_chmod = os.chmod
    stuck = files["backup"]

    def chmod(path, mode, **kw):
        if Path(path) == stuck:
            raise PermissionError("read-only volume")
        real_chmod(path, mode, **kw)

    monkeypatch.setattr(private_files.os, "chmod", chmod)
    report = tighten_private_files(tmp_path)
    assert any(str(stuck) in line for line in report.failed)
    assert _mode(files["learning_restore_point"]) == 0o600


# -- writers ---------------------------------------------------------------------------


@pytest.fixture
def open_umask():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def _store(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
    conn.close()
    os.chmod(path, 0o644)
    return path


def test_routine_backups_are_owner_only(tmp_path, open_umask) -> None:
    from superlocalmemory.infra.backup import BackupManager

    _store(tmp_path / "memory.db")
    _store(tmp_path / "learning.db")
    manager = BackupManager(db_path=tmp_path / "memory.db", base_dir=tmp_path)
    name = manager.create_backup()
    assert name
    written = sorted(manager.backup_dir.glob("*.db"))
    assert written, "no backup was written"
    for path in written:
        assert _mode(path) == 0o600, f"{path.name} is {oct(_mode(path))}"
    assert _mode(manager.backup_dir) == 0o700


def test_backup_sets_are_owner_only(tmp_path, open_umask) -> None:
    from superlocalmemory.infra.backup import BackupCoordinator

    _store(tmp_path / "memory.db")
    coordinator = BackupCoordinator(("memory.db",), tmp_path, tmp_path / "backups")
    manifest = coordinator.create_backup_set()
    folder = tmp_path / "backups" / f"backup_{manifest.set_id}"
    assert _mode(folder) == 0o700
    for path in folder.iterdir():
        assert _mode(path) == 0o600, f"{path.name} is {oct(_mode(path))}"


def test_copy_private_never_takes_the_source_mode(tmp_path, open_umask) -> None:
    source = _open_file(tmp_path / "learning.db", 0o644, b"learned")
    copy = tmp_path / "learning.db.backup_1"
    private_files.copy_private(source, copy)
    assert copy.read_bytes() == b"learned"
    assert _mode(copy) == 0o600


def test_dashboard_learning_backup_is_owner_only() -> None:
    text = (REPO / "src/superlocalmemory/server/routes/learning.py").read_text(encoding="utf-8")
    body = text[text.index("def learning_backup"):text.index("def learning_backup") + 900]
    assert "copy_private(" in body and "copy2(" not in body


def test_damaged_store_copies_are_owner_only(tmp_path, open_umask) -> None:
    from superlocalmemory.storage import _restore_damaged

    target = _open_file(tmp_path / "memory.db", 0o644, b"damaged bytes")
    _open_file(tmp_path / "memory.db-wal", 0o644, b"log")
    safety = _restore_damaged._keep_damaged(target)
    assert safety.read_bytes() == b"damaged bytes"
    assert _mode(safety) == 0o600
    assert _mode(Path(f"{safety}-wal")) == 0o600
    assert _mode(safety.parent) == 0o700

    staged = _store(tmp_path / "staged.db")
    _restore_damaged._replace_file(staged, target)
    assert _mode(target) == 0o600, "the restored store must be as private as before"


def test_restore_safety_folder_is_owner_only(tmp_path, open_umask) -> None:
    from superlocalmemory.storage.backup import restore_pre_migration_snapshot

    live = _store(tmp_path / "data" / "memory.db")
    snapshot = _store(tmp_path / "snap" / "memory-20260901-101010-pre-migration.db")
    safety = restore_pre_migration_snapshot(snapshot, live)
    assert _mode(safety) == 0o600
    assert _mode(safety.parent) == 0o700


def test_the_daemon_runs_the_pass_before_migrations() -> None:
    source = (REPO / "src/superlocalmemory/server/unified_daemon.py").read_text(encoding="utf-8")
    call = source.index("tighten_private_files(_private_root)")
    assert call < source.index("_result = apply_all(_learning_db, _memory_db)")
    assert source.index("perform_pending_restore(") < call
