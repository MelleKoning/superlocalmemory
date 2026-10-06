# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The start-up check of confirmed kinds, run in the background.

``reconcile_confirmed`` repairs ``fact_type`` on confirmed-kind rows that a
4.1.18 downgrade window left out of step. Finding them means reading the tail
of every confirmed fact row (the kind columns sit after the vector), which
took 22 s on a 2 GB store - and 4.1.21 did it on the start-up path, inside a
write transaction, on every start of every process that opened the store: the
daemon was not ready and nothing could be saved until it finished.

Now:

* it runs on its own thread once the engine is up, reading with no write lock
  and repairing in short, paced write transactions;
* only one process at a time runs it for a profile (a file lock next to the
  store); another engine that starts meanwhile reports ``observed``;
* a finished pass is recorded next to the store, and the check is not repeated
  for a day while the same version keeps running - drift only comes from an
  older version writing to the store, and any other version starting changes
  the ``.last_version`` fingerprint the record is tied to;
* ``stop()`` (engine close) interrupts the read and is joined within the
  close's own budget, so no thread keeps touching a closed store;
* its state is reported truthfully on the memory-kinds status: ``pending``,
  ``running``, ``complete`` (with how many rows it repaired), ``up_to_date``,
  ``observed``, ``stopped`` or ``failed`` (with why). A failure is a warning,
  never a failed start, exactly as before.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

THREAD_NAME = "slm-kind-reconcile"

#: How long a finished pass stands while the same version keeps running.
RECHECK_AFTER = timedelta(hours=24)


def _store_dir(db: Any) -> Path | None:
    path = getattr(db, "db_path", None)
    if path is None or str(path) == ":memory:":
        return None
    return Path(path).parent


def _profile_key(profile_id: str) -> str:
    return hashlib.sha256(str(profile_id).encode("utf-8")).hexdigest()[:16]


def _fingerprint(store_dir: Path) -> dict[str, Any]:
    from superlocalmemory.storage._downgrade import _last_version_fingerprint
    from superlocalmemory.storage._snapshot_manifest import package_version

    return {"package": package_version(), **_last_version_fingerprint(store_dir)}


class KindReconcileTask:
    """One background pass of the confirmed-kind check for one profile."""

    def __init__(self, db: Any, profile_id: str, *, recheck_after: timedelta = RECHECK_AFTER,
                 pause_seconds: float = 0.05) -> None:
        self._db, self._profile_id = db, profile_id
        self._recheck_after, self._pause = recheck_after, pause_seconds
        self._done, self._stop = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {"state": "pending", "fixed": 0, "seconds": None,
                                       "started_at": None, "finished_at": None, "error": None}

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> "KindReconcileTask":
        self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[threading.Thread]:
        """Ask the pass to stop; return its thread for the caller's bounded join."""
        self._stop.set()
        thread = self._thread
        return [thread] if thread is not None and thread.is_alive() else []

    def wait(self, timeout: float | None = None) -> bool:
        """True once the pass has finished (in any final state)."""
        return self._done.wait(timeout)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    # -- the pass --------------------------------------------------------------

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self._state = {**self._state, **changes}

    def _marker_path(self, store_dir: Path) -> Path:
        return store_dir / f".kind-reconcile-{_profile_key(self._profile_id)}.json"

    def _recent_pass(self, store_dir: Path | None) -> dict[str, Any] | None:
        if store_dir is None:
            return None
        from superlocalmemory.storage._durable_json import read_json

        marker = read_json(self._marker_path(store_dir))
        if not marker or marker.get("profile_id") != self._profile_id:
            return None
        if marker.get("fingerprint") != _fingerprint(store_dir):
            return None
        try:
            at = datetime.fromisoformat(str(marker.get("finished_at")))
        except ValueError:
            return None
        return marker if datetime.now(UTC) - at < self._recheck_after else None

    def _record_pass(self, store_dir: Path | None, fixed: int) -> None:
        if store_dir is None:
            return
        from superlocalmemory.storage._durable_json import write_json_atomic

        try:
            write_json_atomic(self._marker_path(store_dir), {
                "profile_id": self._profile_id, "fixed": fixed,
                "finished_at": datetime.now(UTC).isoformat(),
                "fingerprint": _fingerprint(store_dir)})
        except OSError as exc:  # the next start simply checks again
            logger.debug("kind reconcile record not written: %s", exc)

    def _run(self) -> None:
        started = time.perf_counter()
        self._set(state="running", started_at=datetime.now(UTC).isoformat())
        try:
            self._run_locked()
        except Exception as exc:  # noqa: BLE001 - a warning, never a failed start
            from superlocalmemory.storage.memory_kind_writes import ReconcileStopped

            if isinstance(exc, ReconcileStopped) or self._stop.is_set():
                self._set(state="stopped")
            else:
                logger.warning("Memory kind reconcile skipped at start: %s", exc)
                self._set(state="failed", error=type(exc).__name__)
        finally:
            self._set(seconds=round(time.perf_counter() - started, 3),
                      finished_at=datetime.now(UTC).isoformat())
            self._done.set()

    def _run_locked(self) -> None:
        from superlocalmemory.core.file_lock import LockHeldError, exclusive_lock
        from superlocalmemory.storage.memory_kind_writes import reconcile_confirmed_pass

        store_dir = _store_dir(self._db)
        recent = self._recent_pass(store_dir)
        if recent is not None:
            self._set(state="up_to_date", checked_at=recent.get("finished_at"))
            return
        lock_path = (store_dir / f".kind-reconcile-{_profile_key(self._profile_id)}.lock"
                     if store_dir is not None else None)
        try:
            if lock_path is None:
                result = self._pass(reconcile_confirmed_pass)
            else:
                with exclusive_lock(lock_path, timeout_s=0.0):
                    result = self._pass(reconcile_confirmed_pass)
        except LockHeldError:
            self._set(state="observed")  # another engine is running this same check
            return
        self._set(state="complete", fixed=result.fixed, drifted=result.drifted,
                  all_checked=result.complete, max_hold_ms=result.max_hold_ms)
        if result.fixed:
            logger.info("Repaired the stored type of %d confirmed memories", result.fixed)
        if result.complete:
            self._record_pass(store_dir, result.fixed)

    def _pass(self, reconcile_confirmed_pass: Any) -> Any:
        return reconcile_confirmed_pass(self._db, self._profile_id,
                                        pause_seconds=self._pause,
                                        should_stop=self._stop.is_set)


__all__ = ["KindReconcileTask", "RECHECK_AFTER", "THREAD_NAME"]
