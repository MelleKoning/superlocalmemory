# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for core.safe_fs."""

from __future__ import annotations

import os
import stat
import sqlite3
import sys
from pathlib import Path

import pytest


def _import_module():
    import superlocalmemory.core.safe_fs as sf
    return sf


# ---------------------------------------------------------------------------
# _safe_open_db
# ---------------------------------------------------------------------------

def test_safe_open_creates_db_with_mode_0600(tmp_path: Path) -> None:
    sf = _import_module()
    db = tmp_path / "t.db"
    conn = sf._safe_open_db(db)
    try:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        cur = conn.execute("SELECT x FROM t")
        assert cur.fetchone()[0] == 1
    finally:
        conn.close()
    assert db.exists()
    st = db.stat()
    if os.name != "nt":  # POSIX mode bits; Windows has none
        assert stat.S_IMODE(st.st_mode) == 0o600, f"Mode is {oct(st.st_mode)}"


def test_safe_open_preserves_wal_mode(tmp_path: Path) -> None:
    sf = _import_module()
    db = tmp_path / "t.db"
    conn = sf._safe_open_db(db)
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        assert mode.lower() == "wal"
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
        conn.commit()
    finally:
        conn.close()
    # Second connection reads the same row (proves WAL aux files work)
    conn2 = sf._safe_open_db(db)
    try:
        assert conn2.execute("SELECT x FROM t").fetchone()[0] == 42
    finally:
        conn2.close()


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_safe_open_refuses_symlink(tmp_path: Path) -> None:
    sf = _import_module()
    real = tmp_path / "real.db"
    real.write_bytes(b"")
    link = tmp_path / "link.db"
    link.symlink_to(real)
    with pytest.raises(sf.SafeFsError):
        sf._safe_open_db(link)


def test_safe_open_auto_tightens_parent_dir(tmp_path: Path) -> None:
    sf = _import_module()
    # Parent dir is world-readable initially
    os.chmod(tmp_path, 0o755)
    db = tmp_path / "t.db"
    conn = sf._safe_open_db(db)
    conn.close()
    # After open, parent must be 0700
    if os.name != "nt":  # POSIX mode bits; Windows has none
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


# ---------------------------------------------------------------------------
# validate_data_dir — cloud-sync detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_component", [
    "iCloud Drive",
    "Mobile Documents",
    "CloudDocs",
    "Dropbox",
    "Google Drive",
    "OneDrive",
    "Box Sync",
])
def test_validate_data_dir_refuses_cloud_sync(tmp_path: Path, bad_component: str) -> None:
    sf = _import_module()
    bad = tmp_path / bad_component / "slm"
    bad.mkdir(parents=True)
    with pytest.raises(sf.SafeFsError, match="cloud|sync|iCloud|Drive|Dropbox|OneDrive|Box"):
        sf.validate_data_dir(bad)


def test_validate_data_dir_accepts_local_path(tmp_path: Path) -> None:
    sf = _import_module()
    sf.validate_data_dir(tmp_path)  # no raise


def test_validate_data_dir_error_mentions_remediation(tmp_path: Path) -> None:
    sf = _import_module()
    bad = tmp_path / "Dropbox" / "slm"
    bad.mkdir(parents=True)
    with pytest.raises(sf.SafeFsError) as excinfo:
        sf.validate_data_dir(bad)
    msg = str(excinfo.value)
    assert "SLM_DATA_DIR" in msg or "data_dir" in msg.lower()
