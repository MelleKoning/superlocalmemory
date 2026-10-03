# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Shared fixture: an isolated credential store for backup-key tests.

The real OS keychain is never touched: ``keyring`` is made unimportable, so
``cloud_backup._store_credential`` uses its 0600 file fallback inside the
test's own data root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture
def key_env(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "keyring", None)
    data = Path(tmp_path) / "slm-data"
    data.mkdir()
    import superlocalmemory.infra.cloud_backup as cb

    monkeypatch.setattr(cb, "MEMORY_DIR", data)
    monkeypatch.setattr(cb, "DB_PATH", data / "memory.db")
    return data


def files_containing(root: Path, needles: list[bytes]) -> list[Path]:
    hits = []
    for path in root.rglob("*"):
        if path.is_file():
            blob = path.read_bytes()
            if any(n in blob for n in needles):
                hits.append(path)
    return hits
