# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Restore-point copies are owner-only, like the live store (audit 4.1.20 L9).

The live memory.db is 0600, but its copies were written 0644: every memory
readable by any other account on the machine, one directory over.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

from superlocalmemory.storage import backup

from ._upgrade_store import add_memory, current_store

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_snapshot_files_and_folder_are_owner_only(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["a private fact"])
    snaps = tmp_path / "pre-migration-snapshots"
    old_umask = os.umask(0o022)            # the usual default: files 0644
    try:
        backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)
    finally:
        os.umask(old_umask)

    copies = sorted(snaps.glob("*-pre-migration.db"))
    assert {p.name.split("-")[0] for p in copies} == {"memory", "learning"}
    for path in copies:
        assert _mode(path) == 0o600, f"{path.name} is {oct(_mode(path))}"
    assert _mode(snaps) == 0o700


def test_an_existing_open_folder_is_tightened(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    snaps = tmp_path / "pre-migration-snapshots"
    snaps.mkdir(exist_ok=True)
    os.chmod(snaps, 0o755)
    backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)
    assert _mode(snaps) == 0o700
