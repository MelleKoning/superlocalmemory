# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Restoring over a live store that SQLite can no longer read.

A damaged store is exactly when a person most needs a restore point, so a live
store that cannot be read must never block one (L1-02). Nothing can be carried
across from such a store -- it cannot be compared with the copy -- so the
restore is a full one, and the preview and the CLI say so before anyone
confirms.

The damaged files are kept, byte for byte, in ``pre-restore/`` first: SQLite
cannot read them, so its backup API cannot copy them, but a recovery tool might.
The verified copy is then written in through SQLite when SQLite can still open
the file, and otherwise put in place by an atomic rename. The rename is safe
here because a restore runs only at start-up, holding the writer lease, with no
other daemon alive (``_restore_boot``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from superlocalmemory.infra.private_files import copy_private, make_private_dir
from superlocalmemory.storage._restore_types import SAFETY_DIR

logger = logging.getLogger(__name__)

LIVE_UNREADABLE_WARNING = (
    "Your current memory store cannot be read; it looks damaged. Restoring this copy "
    "replaces it entirely, so nothing written after the copy can be carried over. "
    "The damaged store is kept aside first, in the pre-restore folder.")

_TRANSIENT = ("locked", "busy")
_COMPANIONS = ("-wal", "-shm", "-journal")


def is_unreadable_error(exc: BaseException) -> bool:
    """A SQLite error that means the file is damaged or missing, not merely busy."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    text = str(exc).lower()
    return not any(word in text for word in _TRANSIENT)


def live_store_problem(db: Path) -> str | None:
    """None when ``db`` reads and passes SQLite's quick check; else why not (for logs)."""
    db = Path(db)
    if not db.exists():
        return "the store file is missing"
    try:
        with closing(sqlite3.connect(f"{db.absolute().as_uri()}?mode=ro", uri=True)) as conn:
            result = conn.execute("PRAGMA quick_check").fetchone()[0]
    except sqlite3.Error as exc:
        return f"{type(exc).__name__}: {exc}"
    return None if result == "ok" else f"quick_check: {result}"


def _keep_damaged(target: Path) -> Path:
    """Copy the damaged store and its companions aside, byte for byte."""
    safety_dir = target.parent / SAFETY_DIR
    make_private_dir(safety_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safety = safety_dir / f"{target.stem}-{stamp}-damaged-before-restore{target.suffix}"
    # Copies of the store are owner-only, like the store (infra/private_files).
    if target.exists():
        copy_private(target, safety)
    for suffix in _COMPANIONS[:1]:          # the log may hold the newest pages
        companion = Path(f"{target}{suffix}")
        if companion.exists() and companion.stat().st_size:
            copy_private(companion, Path(f"{safety}{suffix}"))
    return safety


def _replace_file(staged: Path, target: Path) -> None:
    replacement = target.with_name(f"{target.name}.restoring")
    copy_private(staged, replacement)
    for suffix in _COMPANIONS:
        Path(f"{target}{suffix}").unlink(missing_ok=True)
    os.replace(replacement, target)
    try:
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
    except OSError:
        return                                   # Windows: no directory handles
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_over_damaged(staged: Path, target: Path) -> Path:
    """Put the verified ``staged`` copy in place of an unreadable ``target``.

    Returns the path of the byte copy of the damaged store. Raises if the
    result does not pass SQLite's check (the damaged copy is kept either way).
    """
    from superlocalmemory.storage.backup import LiveStoreWriteError, _write_into_live_db

    target = Path(target)
    safety = _keep_damaged(target)
    try:
        if target.exists():
            _write_into_live_db(staged, target)
            if live_store_problem(target) is None:
                return safety
    except (sqlite3.DatabaseError, LiveStoreWriteError) as exc:
        logger.info("[SLM] Writing into the damaged store failed (%s); replacing the file", exc)
    _replace_file(staged, target)
    problem = live_store_problem(target)
    if problem is not None:
        raise RuntimeError(f"the restored store failed its check ({problem})")
    return safety


__all__ = ["LIVE_UNREADABLE_WARNING", "is_unreadable_error", "live_store_problem",
           "write_over_damaged"]
