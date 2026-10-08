# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon reads its store once at start, so the first recalls find it in memory.

Measured on a copy of a 2 GB, 22k-fact store with a cold OS cache: the first
lookup of a popular entity took 0.5-3 s and the 50-newest query 5.8 s, because
each reads rows scattered across the file. Reading the whole file in order
took 354 ms; afterwards the same lookup took 11 ms. Nothing is changed: the
bytes are read and dropped.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from superlocalmemory.storage import store_cache_warm


def _file(path: Path, size: int) -> Path:
    path.write_bytes(b"x" * size)
    return path


def test_the_store_and_its_journal_are_read_in_full(tmp_path: Path) -> None:
    db = _file(tmp_path / "memory.db", 3 * store_cache_warm.CHUNK + 17)
    wal = _file(tmp_path / "memory.db-wal", 1234)
    expected = db.stat().st_size + wal.stat().st_size
    assert store_cache_warm.warm([db, wal], max_bytes=10**9) == expected


def test_a_store_larger_than_the_cap_is_not_read(tmp_path: Path) -> None:
    big = _file(tmp_path / "memory.db", 4096)
    small = _file(tmp_path / "memory.db-wal", 100)
    assert store_cache_warm.warm([big, small], max_bytes=1000) == 100


def test_a_missing_journal_is_skipped(tmp_path: Path) -> None:
    db = _file(tmp_path / "memory.db", 500)
    assert store_cache_warm.warm([db, tmp_path / "memory.db-wal"], max_bytes=10**9) == 500


def test_the_cap_follows_this_computer_s_memory(tmp_path: Path, monkeypatch) -> None:
    db = _file(tmp_path / "memory.db", 2048)
    engine = SimpleNamespace(_db=SimpleNamespace(db_path=db))
    monkeypatch.setattr(store_cache_warm, "total_ram_gb", lambda: 16.0)
    assert store_cache_warm.warm_engine_store(engine) == 2048
    monkeypatch.setattr(store_cache_warm, "total_ram_gb", lambda: 0.0)  # unknown: do nothing
    assert store_cache_warm.warm_engine_store(engine) == 0
    assert store_cache_warm.warm_engine_store(SimpleNamespace(_db=None)) == 0


_OPENED_HERE: list[str] = []
_WATCH: dict[str, str] = {}


def _record_opens(event: str, args: tuple) -> None:
    root = _WATCH.get("root")
    if root and event == "open" and args and str(args[0]).startswith(root):
        _OPENED_HERE.append(str(args[0]))


def test_warming_never_opens_the_store_inside_the_daemon(tmp_path: Path, monkeypatch) -> None:
    """The daemon holds SQLite connections to this store. Opening and closing
    the same files with plain file handles in that process interferes with
    SQLite's file locks; on a fresh store a save was acknowledged and then lost
    with "disk I/O error". The read must happen in a separate process, which
    warms the shared operating-system cache just the same."""
    import sys

    db = _file(tmp_path / "memory.db", 2 * store_cache_warm.CHUNK + 5)
    wal = _file(tmp_path / "memory.db-wal", 777)
    engine = SimpleNamespace(_db=SimpleNamespace(db_path=db))
    monkeypatch.setattr(store_cache_warm, "total_ram_gb", lambda: 16.0)
    sys.addaudithook(_record_opens)
    _OPENED_HERE.clear()
    _WATCH["root"] = str(tmp_path)
    try:
        read = store_cache_warm.warm_engine_store(engine)
    finally:
        _WATCH.pop("root", None)
    assert _OPENED_HERE == []
    assert read == db.stat().st_size + wal.stat().st_size


def test_the_daemon_starts_it() -> None:
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.lifespan)
    assert "store_cache_warm.start(engine)" in source
