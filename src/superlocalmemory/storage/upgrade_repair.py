# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The one-time data repair, run once on upgrade -- and only with a way back.

``storage/store_repair.py`` undoes damage earlier builds left (memories whose
text was blanked, "replaced" marks an automatic check got wrong, empty entities
created by questions). It changes rows, so it runs only when ALL of these hold:

  1. it has not completed before on this data directory (``upgrade-repair.json``);
  2. the update's own schema change (M052) is recorded complete -- the repair
     runs after it, never against a half-migrated store;
  3. a safety copy taken BEFORE that change exists and verifies now (its
     checksum from the manifest, SQLite's own check). No verified copy, no
     repair: the user must always be able to restore the unrepaired store.

The counts are written to ``upgrade-repair.json`` and into that copy's
manifest, so the restore-point preview can say what restoring it would undo.

Call it once per start after ``apply_all`` and before the engine opens (the
daemon lifespan; ``slm db migrate`` may call it too). A restore does not reset
the record: a user who restores to before the update is not repaired again
behind their back.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic
from superlocalmemory.storage._restore_types import REPAIR_RECORD, SNAPSHOTS_DIR

logger = logging.getLogger(__name__)

REPAIR_ID = "store-repair-4.1.19"
#: The update this repair belongs to: copies taken before this schema version
#: hold the store as it was before the update (and so before the repair).
UPGRADE_SCHEMA = 52
_UPGRADE_MIGRATION = "M052_memory_kinds"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _update_applied(memory_db: Path) -> bool:
    try:
        with closing(sqlite3.connect(f"{memory_db.absolute().as_uri()}?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT status FROM migration_log WHERE name=?",
                               (_UPGRADE_MIGRATION,)).fetchone()
    except sqlite3.Error:
        return False
    return bool(row) and row[0] == "complete"


def _verified_pre_upgrade_point(data_root: Path) -> Any | None:
    """Newest copy with a manifest, taken before the update, that verifies now."""
    from superlocalmemory.storage import _snapshot_manifest as sm
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage.backup import SnapshotUnusableError

    snaps = Path(data_root) / SNAPSHOTS_DIR
    for point in ur.list_restore_points(data_root):
        if point.legacy:
            continue
        manifest = sm.read_manifest(snaps, point.point_id) or {}
        if int(manifest.get("schema_version_before") or 0) >= UPGRADE_SCHEMA:
            continue
        try:
            ur.verify_point(point)
        except SnapshotUnusableError as exc:
            logger.warning("[SLM] Safety copy %s does not verify (%s); not repairing "
                           "on top of it", point.point_id, exc)
            continue
        return point
    return None


def run_store_repair_once(data_root: Path, memory_db: Path) -> dict[str, Any]:
    """Run the repair if, and only if, it is safe and has not run. Never raises
    for a store that cannot be repaired; the returned ``status`` says why."""
    from superlocalmemory.storage import _snapshot_manifest as sm
    from superlocalmemory.storage.store_repair import apply_repair, plan_repair
    from superlocalmemory.storage.write_lock import get_write_lock

    data_root, memory_db = Path(data_root), Path(memory_db)
    record_path = data_root / REPAIR_RECORD
    record = read_json(record_path) or {}
    if record.get("status") == "complete":
        return {**record, "status": "already_done"}
    if not memory_db.exists() or not _update_applied(memory_db):
        return {"status": "waiting_for_update"}
    point = _verified_pre_upgrade_point(data_root)
    if point is None:
        logger.info("[SLM] One-time repair waits: no verified copy from before the update")
        return {"status": "waiting_for_snapshot"}

    with get_write_lock(memory_db), closing(
            sqlite3.connect(str(memory_db), isolation_level=None, timeout=30)) as conn:
        plan = plan_repair(conn)
        planned = record.get("planned") or plan.summary()
        write_json_atomic(record_path, {"repair_id": REPAIR_ID, "status": "started",
                                        "point_id": point.point_id, "planned": planned,
                                        "started_at": record.get("started_at") or _now()})
        result = apply_repair(conn, plan)
    done = {"repair_id": REPAIR_ID, "status": "complete", "point_id": point.point_id,
            "planned": planned, "result": asdict(result), "completed_at": _now()}
    write_json_atomic(record_path, done)
    sm.update_manifest(data_root / SNAPSHOTS_DIR, point.point_id, {"repair": {
        "repair_id": REPAIR_ID, "completed_at": done["completed_at"], **done["result"]}})
    logger.info("[SLM] One-time repair after the update: %s (copy kept: %s)",
                done["result"], point.point_id)
    return {**done, "status": "ran"}


__all__ = ["REPAIR_ID", "run_store_repair_once"]
