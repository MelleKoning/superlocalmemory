# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A learning.db with the saved-views table, made by the real migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.storage.migrations import M054_saved_views as M054


def learning_db(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "learning.db"
    with sqlite3.connect(str(path)) as conn:
        M054.repair(conn)
    return path
