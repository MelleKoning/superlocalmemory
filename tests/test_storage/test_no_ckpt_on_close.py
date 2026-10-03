"""Regression tests for the WAL close-path deadlock fix (NO_CKPT_ON_CLOSE).

Root cause (production incident + in-suite reproduction, 2026-08-13):
closing a WAL-mode connection triggers a checkpoint inside sqlite3WalClose.
That checkpoint can wait indefinitely on reader marks pinned by another
connection — while the close path holds SQLite's process-global VFS mutex,
so every later sqlite3.connect() in the process convoys behind it.
PRAGMA busy_timeout does NOT apply to the close path.

Fix: every per-call connection opened by DatabaseManager._connect() sets
SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, so close() never checkpoints and cannot
block. Routine checkpointing continues via PRAGMA wal_autocheckpoint=400.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage.database import DatabaseManager


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "setconfig"),
    reason="Connection.setconfig (Python 3.12+) is needed to turn off the close-time "
    "checkpoint; on 3.11 SLM warns once and runs without it",
)
def test_close_leaves_wal_uncheckpointed(tmp_path) -> None:
    """With NO_CKPT_ON_CLOSE, close() must not checkpoint the WAL.

    Observable contract: after writes through the per-call connection model,
    the ``-wal`` sidecar still holds the frames (a checkpointing last-close
    would merge them into the main database file and remove/truncate the
    sidecar). This is the property that makes close() non-blocking.
    """
    db = tmp_path / "memory.db"
    mgr = DatabaseManager(db)
    mgr.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
    mgr.execute("INSERT INTO t (id) VALUES (1)")

    wal = tmp_path / "memory.db-wal"
    assert wal.exists() and wal.stat().st_size > 0, (
        "WAL sidecar missing/empty after close — conn.close() checkpointed, "
        "which is exactly the blocking path behind the WAL close deadlock. "
        "DatabaseManager._connect() must set SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE."
    )


def test_write_and_close_roundtrip(tmp_path) -> None:
    """Durability is unaffected: written rows survive connection close."""
    db = tmp_path / "memory.db"
    mgr = DatabaseManager(db)
    mgr.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
    mgr.execute("INSERT INTO t (id) VALUES (42)")

    probe = sqlite3.connect(str(db))
    try:
        assert probe.execute("SELECT id FROM t").fetchone() == (42,)
    finally:
        probe.close()
