# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Safety copies named the older way are pruned like any other generation.

The owner's snapshot folder held ``memory-20260823-144350-pre-4.1.0.db`` and
``memory-20260823-175042-pre-4.1.2.db``, each with ``-shm`` and ``-wal`` beside
it: 1.28 GB that retention never saw, because it only recognised names ending
``-pre-migration.db``. Those copies stay forever on every machine that has them.

What retention must do with them, and must never do:

  * count them as generations, ordered by the time in their name, under the
    same "keep the two newest" rule, and take their side files with them;
  * keep the newest generations whichever naming they use;
  * delete only by explicit full path, only directly inside the folder, and
    never through a symlink.

The older names carry no time zone (they were written in local time; the newer
ones are UTC), so a legacy copy whose time is within a day of a newer one might
be either side of it. Retention keeps such a copy rather than guess.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.backup import _gc_old_backups


def _db(path: Path) -> Path:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
        conn.commit()
    finally:
        conn.close()
    return path


def _legacy(root: Path, stem: str, stamp: str, version: str) -> list[Path]:
    """A legacy copy and the side files SQLite left beside it."""
    main = _db(root / f"{stem}-{stamp}-pre-{version}.db")
    wal = root / f"{main.name}-wal"
    shm = root / f"{main.name}-shm"
    wal.write_bytes(b"\0" * 64)
    shm.write_bytes(b"\0" * 32)
    return [main, wal, shm]


def _current(root: Path, stamp: str) -> list[Path]:
    return [
        _db(root / f"{stem}-{stamp}-pre-migration.db")
        for stem in ("memory", "learning")
    ]


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "pre-migration-snapshots"
    folder.mkdir()
    return folder


def _names(root: Path) -> set[str]:
    return {p.name for p in root.iterdir()}


class TestLegacyNamesAreGenerations:
    def test_the_owners_folder_is_pruned_to_the_two_newest(self, root: Path) -> None:
        """The exact names found on the owner's machine."""
        old_a = _legacy(root, "memory", "20260823-144350", "4.1.0")
        old_b = _legacy(root, "memory", "20260823-175042", "4.1.2")
        new_a = _current(root, "20260912-101500-123456")
        new_b = _current(root, "20260912-101504-654321")

        _gc_old_backups(root, keep=2)

        left = _names(root)
        for path in (*old_a, *old_b):
            assert path.name not in left, f"{path.name} was never pruned"
        for path in (*new_a, *new_b):
            assert path.name in left, f"{path.name} is among the newest two"

    def test_the_newest_survive_whichever_naming_they_use(self, root: Path) -> None:
        oldest = _current(root, "20260901-080000-000001")
        middle = _current(root, "20260905-080000-000001")
        newest = _legacy(root, "memory", "20260920-090000", "4.1.17")

        _gc_old_backups(root, keep=2)

        left = _names(root)
        assert all(p.name not in left for p in oldest)
        assert all(p.name in left for p in middle)
        assert all(p.name in left for p in newest), (
            "the newest copy, and its side files, must survive"
        )

    def test_a_legacy_pair_is_one_generation(self, root: Path) -> None:
        """memory and learning copied together are kept or pruned together."""
        oldest = _current(root, "20260801-080000-000001")
        middle = _current(root, "20260901-080000-000001")
        pair = [
            *_legacy(root, "memory", "20260920-090000", "4.1.17"),
            *_legacy(root, "learning", "20260920-090000", "4.1.17"),
        ]

        _gc_old_backups(root, keep=2)

        left = _names(root)
        assert all(p.name not in left for p in oldest)
        assert all(p.name in left for p in middle), (
            "the legacy pair was counted as two generations"
        )
        assert all(p.name in left for p in pair)

    def test_older_names_without_microseconds_are_ordered_too(self, root: Path) -> None:
        old = _db(root / "memory-20260819-110000-pre-migration.db")
        mid = _db(root / "memory-20260820-110000-000001-pre-migration.db")
        new = _db(root / "memory-20260821-110000-pre-migration.db")

        _gc_old_backups(root, keep=2)

        assert not old.exists()
        assert mid.exists() and new.exists()


class TestNothingIsRiskedOnAGuess:
    def test_a_legacy_copy_that_might_be_newer_is_kept(self, root: Path) -> None:
        """Local time, no zone: 09:00 could be later than 10:00 UTC."""
        _current(root, "20260801-080000-000001")
        near = _legacy(root, "memory", "20260912-090000", "4.1.16")
        _current(root, "20260912-100000-000001")
        _current(root, "20260912-100004-000001")

        _gc_old_backups(root, keep=2)

        left = _names(root)
        assert all(p.name in left for p in near), (
            "a copy that may be one of the newest two was deleted on a guess"
        )
        assert "memory-20260801-080000-000001-pre-migration.db" not in left

    def test_a_name_with_no_readable_time_is_never_deleted(self, root: Path) -> None:
        unreadable = _db(root / "0000-pre-migration.db")
        for day in ("01", "02", "03"):
            _current(root, f"202609{day}-080000-000001")

        _gc_old_backups(root, keep=2)

        assert unreadable.exists()

    def test_unrelated_files_are_untouched(self, root: Path) -> None:
        keep = [
            _db(root / "memory.db"),
            _db(root / "memory-20260101-000000-pre-4.0.0.db.bak"),
            _db(root / "memory-pre-4.0.0.db"),
        ]
        (root / "notes-20260101-000000-pre-4.0.0.txt").write_text("mine", encoding="utf-8")
        for day in ("01", "02", "03"):
            _current(root, f"202609{day}-080000-000001")

        _gc_old_backups(root, keep=2)

        for path in keep:
            assert path.exists(), f"{path.name} is not a safety copy"
        assert (root / "notes-20260101-000000-pre-4.0.0.txt").exists()


class TestDeletionStaysInsideTheFolder:
    def test_a_symlink_is_never_followed_or_removed(
        self, root: Path, tmp_path: Path,
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        precious = _db(outside / "precious.db")
        links = [
            root / "memory-20250101-000000-000001-pre-migration.db",
            root / "memory-20250101-000000-pre-4.0.0.db",
            root / "memory-20250101-000000-pre-4.0.0.db-wal",
        ]
        for link in links:
            os.symlink(precious, link)
        for day in ("01", "02", "03"):
            _current(root, f"202609{day}-080000-000001")

        _gc_old_backups(root, keep=2)

        assert precious.exists()
        for link in links:
            assert link.is_symlink(), f"{link.name} was removed through GC"

    def test_nothing_below_the_folder_is_touched(self, root: Path) -> None:
        nested = root / "kept-by-hand"
        nested.mkdir()
        inner = _legacy(nested, "memory", "20250101-000000", "4.0.0")
        for day in ("01", "02", "03"):
            _current(root, f"202609{day}-080000-000001")

        _gc_old_backups(root, keep=2)

        assert all(p.exists() for p in inner)

    def test_a_symlinked_folder_is_left_alone(self, tmp_path: Path) -> None:
        real = tmp_path / "elsewhere"
        real.mkdir()
        for day in ("01", "02", "03"):
            _current(real, f"202609{day}-080000-000001")
        link = tmp_path / "pre-migration-snapshots"
        os.symlink(real, link, target_is_directory=True)

        _gc_old_backups(link, keep=2)

        assert len(list(real.iterdir())) == 6

    def test_side_files_left_without_their_copy_are_removed(self, root: Path) -> None:
        """A prune interrupted after the main file leaves side files behind."""
        orphans = [
            root / "memory-20250101-000000-pre-4.0.0.db-wal",
            root / "memory-20250101-000000-pre-4.0.0.db-shm",
            root / "memory-20250101-000000-000001-pre-migration.db-wal",
        ]
        for orphan in orphans:
            orphan.write_bytes(b"\0" * 16)
        for day in ("01", "02"):
            _current(root, f"202609{day}-080000-000001")

        _gc_old_backups(root, keep=2)

        left = _names(root)
        assert all(o.name not in left for o in orphans)
