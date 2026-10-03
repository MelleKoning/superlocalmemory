# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What restoring a point would change, in counts and plain sentences. Reads only.

``problems`` block a restore (the copy is damaged or too new, the disk is too
full). ``warnings`` do not; they say what the person is about to accept:

  * rows missing now with no recorded deletion come back (L1-01),
  * a damaged live store is replaced entirely, nothing carried over (L1-02),
  * learning recorded since the copy is not carried over (L1-05).

Every message is plain words with no file path and no exception text (L3-21);
the details go to the log.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from superlocalmemory.storage._restore_damaged import LIVE_UNREADABLE_WARNING
from superlocalmemory.storage._restore_types import RestorePoint, RestorePreview

logger = logging.getLogger(__name__)

COUNT_KEYS = (
    "facts_in_snapshot", "facts_now", "memories_in_snapshot", "memories_now",
    "added_memories", "deleted_facts", "deleted_memories", "kind_edits",
    "corrections_lost", "erasures", "returning_facts", "returning_memories",
)


def copy_schema_version(point_files: list[Path]) -> int | None:
    """The highest store version stamped in any file of a copy (None: unstamped)."""
    found: list[int] = []
    for path in point_files:
        try:
            with closing(sqlite3.connect(f"{Path(path).absolute().as_uri()}"
                                         "?mode=ro&immutable=1", uri=True)) as conn:
                row = conn.execute(
                    "SELECT version FROM slm_schema_version WHERE id = 1").fetchone()
        except sqlite3.Error:
            continue
        if row and row[0] is not None:
            found.append(int(row[0]))
    return max(found) if found else None


def not_restorable_reason(schema_version: int | None) -> str | None:
    from superlocalmemory.storage._schema_version import SUPPORTED_SCHEMA_VERSION

    if schema_version is None or schema_version <= SUPPORTED_SCHEMA_VERSION:
        return None
    return (f"This copy was made by a newer version of SuperLocalMemory (store version "
            f"{schema_version}); this version understands up to {SUPPORTED_SCHEMA_VERSION}. "
            "Install the newer version to restore it. Nothing was changed.")


def _returning_warning(counts: dict[str, int]) -> str | None:
    facts, memories = counts["returning_facts"], counts["returning_memories"]
    if not facts and not memories:
        return None
    return (f"{facts:,} memories ({memories:,} records) are missing from your store now, "
            "but no deletion was recorded for them (no forget, no erasure). The restore "
            "brings them back.")


def _compare(memory_db: Path, point: RestorePoint, counts: dict[str, int]) -> bool:
    """Fill ``counts`` from the live store and the copy. False: the live store is unreadable."""
    from superlocalmemory.storage._restore_damaged import is_unreadable_error
    from superlocalmemory.storage._restore_delta import delta_counts, open_compare

    try:
        with closing(open_compare(Path(memory_db), point.memory_snapshot)) as conn:
            counts.update(delta_counts(conn))
        return True
    except sqlite3.Error as exc:
        if not is_unreadable_error(exc):
            raise
        logger.warning("[SLM] The live memory store could not be read for a restore "
                       "preview (%s: %s)", type(exc).__name__, exc)
    from superlocalmemory.storage import _snapshot_manifest as sm

    snap = sm.store_counts(point.memory_snapshot)
    counts.update(facts_in_snapshot=snap["facts"] or 0,
                  memories_in_snapshot=snap["memories"] or 0)
    return False


def build_preview(point: RestorePoint, *, data_root: Path, memory_db: Path) -> RestorePreview:
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage._restore_learning import lost_learning_warning, records_since
    from superlocalmemory.storage.backup import SnapshotUnusableError

    learning_db = Path(memory_db).parent / "learning.db"
    problems: list[str] = []
    warnings: list[str] = []
    counts: dict[str, Any] = dict.fromkeys(COUNT_KEYS, 0)
    verified, live_ok, learning_lost = True, True, 0
    if not point.restorable and point.not_restorable_reason:
        problems.append(point.not_restorable_reason)
    try:
        ur.verify_point(point)
    except SnapshotUnusableError as exc:
        verified = False
        logger.warning("[SLM] Restore point %s failed verification: %s", point.point_id, exc)
        problems.append("This copy is damaged or changed after it was taken, so it cannot "
                        f"be used ({exc}).")
    if verified and point.restorable:
        live_ok = _compare(Path(memory_db), point, counts)
        if not live_ok:
            warnings.append(LIVE_UNREADABLE_WARNING)
        else:
            warnings += [w for w in (_returning_warning(counts),) if w]
        if point.learning_snapshot is not None:
            learning_lost = records_since(learning_db, point.learning_snapshot)
            warnings += [w for w in (lost_learning_warning(learning_lost),) if w]
    needed = ur.disk_needed(point, Path(memory_db),
                            learning_db if learning_db.exists() else None)
    free = ur.free_bytes(Path(data_root))
    if free < needed:
        problems.append(f"Not enough free disk space: the restore needs {needed:,} bytes "
                        f"and {free:,} are free. Nothing has been changed.")
    return RestorePreview(
        point_id=point.point_id, verified=verified, repair=point.repair,
        disk_needed_bytes=needed, disk_free_bytes=free, disk_ok=free >= needed,
        problems=problems, warnings=warnings, restorable=point.restorable,
        live_store_readable=live_ok, learning_records_lost=learning_lost or 0, **counts)


__all__ = ["COUNT_KEYS", "build_preview", "copy_schema_version", "not_restorable_reason"]
