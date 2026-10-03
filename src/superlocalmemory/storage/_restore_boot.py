# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The restore itself: run at start-up, before anything opens the store.

Order, and why each step is where it is:

 1. Refuse while anything else can write: another live daemon (pid file or the
    writer lease's recorded holder), or anyone holding the writer lease -- this
    process included. The lease is then HELD for the whole restore, so nothing
    can start writing part-way through. A refusal keeps the request for the
    next start and changes nothing.
 2. Re-verify the copy (checksum from its manifest, SQLite's own check).
 3. Check the disk can hold a staging copy plus safety copies of the live store.
 4. Export, from the live store as it is NOW, everything that changed since the
    copy, and record in the request that writing has begun and where that
    export is. A restore interrupted after this point re-uses that export: the
    store it would compute a new one from may already be the restored one.
 5. Build a staging copy of the snapshot and apply deletions and erasures to
    it -- erased data never reaches the live store, not even for a moment.
 6. Write the staging copy into the live store through the SQLite backup API,
    one transaction, after a safety copy into ``pre-restore/``. Power lost
    during the write leaves the old store intact (the transaction rolls back)
    and the request in place; the next start runs it again.
 7. Record the outcome, then retire the request (``restore-intent.done-<ts>``).
    Outcome first: a crash between the two re-runs the restore (harmless, the
    same copy is written again with another safety copy) instead of losing the
    note that the newer memories still have to be added back.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic
from superlocalmemory.storage._restore_types import (
    DELTA_DIR, INTENT_NAME, OUTCOME_NAME, SAFETY_DIR, RestoreOutcome,
)

logger = logging.getLogger(__name__)

_STAGING = "staging"
_STAGING_FILES = ("memory.db", "learning.db")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def _other_writer(memory_db: Path) -> str | None:
    """A description of another live process that can write the store, or None."""
    from superlocalmemory.storage import _boot_snapshot
    from superlocalmemory.storage.migration_runner import _foreign_live_daemon

    pid = _foreign_live_daemon(memory_db)
    if pid is not None:
        return f"SuperLocalMemory is still running (process {pid})"
    holder = _boot_snapshot.lease_holder(memory_db)
    if holder is not None:
        who = "another process" if holder == _boot_snapshot.UNKNOWN_HOLDER else f"process {holder}"
        return f"{who} is writing to your memories"
    return None


def _lease_path(memory_db: Path) -> Path:
    resolved = Path(memory_db).expanduser().resolve()
    return resolved.with_name(f"{resolved.name}.writer.lock")


def _write_outcome(data_root: Path, outcome: RestoreOutcome, **extra: Any) -> None:
    write_json_atomic(Path(data_root) / OUTCOME_NAME, {
        **outcome.as_dict(), "at": datetime.now(UTC).isoformat(timespec="seconds"), **extra})


def _retire_intent(data_root: Path, label: str) -> Path | None:
    intent = Path(data_root) / INTENT_NAME
    if not intent.exists():
        return None
    target = Path(data_root) / f"restore-intent.{label}-{_stamp()}.json"
    os.replace(intent, target)
    return target


def _refused(data_root: Path, point_id: str | None, message: str) -> RestoreOutcome:
    outcome = RestoreOutcome(status="refused", point_id=point_id, message=message)
    _write_outcome(data_root, outcome)
    logger.warning("[SLM] Restore not started: %s. Your memories were not changed; "
                   "it will be tried again at the next start.", message)
    return outcome


def _clear_staging(staging_dir: Path, *, remove_dir: bool = False) -> None:
    """Remove this module's own staging files, by explicit path."""
    for name in _STAGING_FILES:
        for suffix in ("", "-wal", "-shm", "-journal"):
            (staging_dir / f"{name}{suffix}").unlink(missing_ok=True)
    if remove_dir:
        try:
            staging_dir.rmdir()          # only if empty
        except OSError:
            pass


def _copy_quiescent(snapshot: Path, dest: Path) -> None:
    """Copy a verified, unchanging snapshot into a new staging file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{dest}{suffix}").unlink(missing_ok=True)
    src = sqlite3.connect(f"{snapshot.absolute().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        with closing(sqlite3.connect(str(dest))) as dst:
            src.backup(dst)
    finally:
        src.close()


def _quiesce(db: Path) -> None:
    """Fold a staging file's log into it so it can be read immutable."""
    with closing(sqlite3.connect(str(db), isolation_level=None)) as conn:
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def perform_pending_restore(
    data_root: Path, memory_db: Path, learning_db: Path, *, config: Any | None = None,
) -> RestoreOutcome | None:
    """Run a requested restore. None when nothing was requested.

    Daemon start step 0 (before migrations and the engine), and ``slm db
    restore`` with the daemon stopped. ``config`` (the loaded settings) lets it
    move vector and graph search back to SQLite until they are rebuilt from the
    restored store; without it ``settle_projections_after_restore`` must be
    called before the backends are promoted again.
    """
    from superlocalmemory.core.file_lock import LockHeldError, exclusive_lock

    data_root, memory_db = Path(data_root), Path(memory_db)
    intent_path = data_root / INTENT_NAME
    if not intent_path.exists():
        return None
    intent = read_json(intent_path)
    if not intent or not intent.get("point_id"):
        _retire_intent(data_root, "invalid")
        outcome = RestoreOutcome(status="invalid", point_id=None,
                                 message="The restore request was unreadable and was set aside.")
        _write_outcome(data_root, outcome)
        return outcome
    point_id = str(intent["point_id"])
    other = _other_writer(memory_db)
    if other:
        return _refused(data_root, point_id, other)
    try:
        lease = exclusive_lock(_lease_path(memory_db), timeout_s=0.0)
        lease.__enter__()
    except LockHeldError:
        return _refused(data_root, point_id, "the memory store is open for writing")
    try:
        return _perform_locked(data_root, memory_db, Path(learning_db), intent, config)
    finally:
        lease.__exit__(None, None, None)


def _perform_locked(data_root: Path, memory_db: Path, learning_db: Path,
                    intent: dict[str, Any], config: Any | None) -> RestoreOutcome:
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage.backup import SnapshotUnusableError

    point_id = str(intent["point_id"])
    try:
        point = ur.find_restore_point(point_id, data_root)
        ur.verify_point(point)
    except (ur.RestoreRefusedError, SnapshotUnusableError) as exc:
        _retire_intent(data_root, "failed")
        outcome = RestoreOutcome(status="snapshot_unusable", point_id=point_id,
                                 message=f"The copy cannot be used ({exc}). Nothing was changed.")
        _write_outcome(data_root, outcome)
        return outcome

    was_writing = intent.get("stage") == "writing"
    try:
        written = _stage_and_write(data_root, memory_db, learning_db, intent, point)
    except Exception as exc:  # noqa: BLE001 - the live store is unchanged (see below)
        logger.exception("[SLM] Restore failed before changing anything")
        _clear_staging(data_root / SAFETY_DIR / _STAGING, remove_dir=True)
        if not was_writing:
            # A first attempt that failed changed nothing: set it aside rather than
            # fail the same way at every start. A request already part-way through
            # stays: an earlier attempt may have written the store, and its export
            # is the only record of what must still be added back.
            _retire_intent(data_root, "failed")
        outcome = RestoreOutcome(status="failed", point_id=point_id,
                                 message=f"The restore could not run ({type(exc).__name__}: "
                                         f"{exc}). Your memories were not changed.")
        _write_outcome(data_root, outcome)
        return outcome
    if isinstance(written, RestoreOutcome):
        return written
    return _finish(data_root, memory_db, learning_db, intent, point, written, config)


def _stage_and_write(data_root: Path, memory_db: Path, learning_db: Path,
                     intent: dict[str, Any], point: Any) -> dict[str, Any] | RestoreOutcome:
    """Export, stage, write the memory store. Anything raised here leaves the live
    store as it was: every step before the write touches only new files, and the
    write is one SQLite transaction that rolls back if it does not finish."""
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage._restore_delta import (
        apply_delta_to_staging, export_delta, open_compare,
    )
    from superlocalmemory.storage.backup import restore_pre_migration_snapshot

    delta = Path(intent["final_delta_dir"]) if intent.get("final_delta_dir") else None
    if intent.get("stage") != "writing" or delta is None or not delta.is_dir():
        needed = ur.disk_needed(point, memory_db, learning_db if learning_db.exists() else None)
        free = ur.free_bytes(data_root)
        if free < needed:
            return _refused(data_root, point.point_id, f"not enough free disk space (needs "
                            f"{needed:,} bytes, {free:,} free)")
        delta = data_root / DELTA_DIR / f"{_stamp()}-final"
        with closing(open_compare(memory_db, point.memory_snapshot)) as conn:
            export_delta(conn, delta)
        intent.update({"stage": "writing", "final_delta_dir": str(delta)})
        write_json_atomic(data_root / INTENT_NAME, intent)

    staging_dir = data_root / SAFETY_DIR / _STAGING
    _clear_staging(staging_dir)
    staged_memory = staging_dir / "memory.db"
    _copy_quiescent(point.memory_snapshot, staged_memory)
    applied = apply_delta_to_staging(staged_memory, delta, data_root=data_root)
    _quiesce(staged_memory)
    staged_learning = None
    if point.learning_snapshot is not None and learning_db.exists():
        staged_learning = staging_dir / "learning.db"
        _copy_quiescent(point.learning_snapshot, staged_learning)
        _quiesce(staged_learning)
    safety = restore_pre_migration_snapshot(staged_memory, memory_db)
    return {"delta": delta, "applied": applied, "safety": [safety],
            "staged_learning": staged_learning}


def _finish(data_root: Path, memory_db: Path, learning_db: Path, intent: dict[str, Any],
            point: Any, written: dict[str, Any], config: Any | None) -> RestoreOutcome:
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage._restore_delta import load_kind_edits, load_memories
    from superlocalmemory.storage.backup import restore_pre_migration_snapshot

    staging_dir = data_root / SAFETY_DIR / _STAGING
    delta, safety = written["delta"], written["safety"]
    if written["staged_learning"] is not None:
        try:
            safety.append(restore_pre_migration_snapshot(written["staged_learning"],
                                                         learning_db))
        except Exception as exc:  # noqa: BLE001 - put the memory store back as it was
            restore_pre_migration_snapshot(safety[0], memory_db)
            _clear_staging(staging_dir, remove_dir=True)
            _retire_intent(data_root, "failed")
            outcome = RestoreOutcome(
                status="failed", point_id=point.point_id,
                safety_copies=[str(p) for p in safety],
                message=f"The restore could not finish ({exc}); your memories were put "
                        "back exactly as they were before it started.")
            _write_outcome(data_root, outcome)
            return outcome
    _clear_staging(staging_dir, remove_dir=True)
    projections = (settle_projections_after_restore(config, data_root)
                   if config is not None else "rebuild_needed")
    pending = bool(intent.get("reimport", True)) and bool(
        load_memories(delta) or load_kind_edits(delta))
    outcome = RestoreOutcome(
        status="restored", point_id=point.point_id, safety_copies=[str(p) for p in safety],
        delta_dir=str(delta), reimport_pending=pending, applied=written["applied"],
        projections=projections,
        message="Your memories were restored to the copy from "
                f"{point.created_at or point.point_id}. A copy of what was replaced is in "
                f"{data_root / SAFETY_DIR}.")
    _write_outcome(data_root, outcome, reimport=None)
    _retire_intent(data_root, "done")
    if intent.get("delta_dir") and Path(intent["delta_dir"]) != delta:
        ur._remove_delta_dir(data_root, intent["delta_dir"])   # superseded by the final export
    logger.info("[SLM] Restored %s from %s (applied %s)", memory_db.name, point.point_id,
                written["applied"])
    return outcome


def settle_projections_after_restore(config: Any, data_root: Path) -> str:
    """Serve search from SQLite until the graph/vector copies are rebuilt.

    The graph and vector projections were built from the store that was just
    replaced. Verifying them needs a stage built from the RESTORED store
    (``ScaleEngineManager.verify(stage_id)`` checks a staged projection against
    the store it was prepared from), so the safe answer is to stop serving the
    stale ones: the state drops to ``local_core`` (SQLite serves every query),
    stages prepared before the restore are retired so they are not resumed, and
    the existing automatic promotion prepares, verifies and promotes a fresh
    projection of the restored store at this or the next start.
    """
    state = str(getattr(config, "scale_engine_state", "") or "local_core").lower()
    if state == "local_core":
        return "serving_from_sqlite"
    try:
        from superlocalmemory.core.scale_engine import ScaleEngineManager

        manager = ScaleEngineManager(config)
        for stage in manager.status().get("stages") or []:
            if str(stage.get("state")) in {"prepared", "verified"} and stage.get("stage_id"):
                manager._retire_superseded_stage(str(stage["stage_id"]))
    except Exception as exc:  # noqa: BLE001 - demotion below still makes it safe
        logger.warning("[SLM] Could not retire projection stages after restore: %s", exc)
    config.scale_engine_state = "local_core"
    config.graph_backend = "auto"
    config.vector_backend = "auto"
    save = getattr(config, "save", None)
    if callable(save):
        save()
    return "rebuild_scheduled"


__all__ = ["perform_pending_restore", "settle_projections_after_restore"]
