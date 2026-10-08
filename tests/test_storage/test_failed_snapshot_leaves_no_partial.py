"""A snapshot whose copy step fails (the source is already damaged, GitHub #153)
leaves nothing behind: before, the staging file stayed as a 0-byte
``*.db.partial`` beside the real snapshots."""
import sqlite3

import pytest

from superlocalmemory.storage import backup


def test_a_copy_that_fails_removes_its_staging_file(tmp_path):
    src = tmp_path / "memory.db"
    src.write_bytes(b"this is not a database" * 100)
    dest = tmp_path / "snapshots" / "memory-pre-migration.db"
    with pytest.raises(sqlite3.DatabaseError):
        backup._backup_via_sqlite_api(src, dest)
    assert list(dest.parent.iterdir()) == []
