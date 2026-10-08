# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The memory-store check after an upgrade, and the repair the owner starts.

Most people never run ``slm db integrity`` or ``slm db repair``. So:

* once after each upgrade, a few minutes after SLM starts, the store is checked
  in the background. The check only reads and counts (``integrity_scan.plan``)
  and writes its result to ``store-check.json`` in the data folder. It never
  changes the store;
* the dashboard's Health page shows that result in plain words with
  **Repair now**. A repair the owner starts first takes a full backup copy
  (``infra.backup.BackupManager``); if the copy fails nothing is changed. It
  then runs the same repair as ``slm db repair --apply``, inside this daemon,
  and checks again.

Both run in one background thread at a time and never raise into the daemon.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic

logger = logging.getLogger(__name__)

STATE_FILE = "store-check.json"
FIRST_CHECK_DELAY_S = 180.0
BACKUP_LABEL = "before-repair"

#: What the dashboard says for each count, in the order it lists them.
FINDINGS: dict[str, str] = {
    "stale_vectors": "search vectors that no longer match their memory",
    "lance_orphans": "LanceDB search entries for memories that are gone",
    "unreachable_vectors": "search vectors no memory uses",
    "leftover_rows": "leftover rows from deleted memories",
    "erased_words": "copies of erased words still stored",
    "unfinished_updates": "index updates that never finished",
    "memories_without_fact": "memories that lost their searchable fact",
}

_busy = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _int(value: Any) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def findings(plan: dict[str, Any]) -> dict[str, int]:
    """Counts a repair would act on, from ``integrity_scan.plan``."""
    parity = plan.get("vector_parity") or {}
    erased = plan.get("erased_text") or {}
    obligations = plan.get("failed_obligations") or {}
    return {
        "stale_vectors": _int(parity.get("stale_vectors")),
        "lance_orphans": _int((parity.get("lance") or {}).get("orphans")),
        "unreachable_vectors": _int(plan.get("unreachable_vectors")),
        "leftover_rows": sum(_int(o.get("rows")) for o in plan.get("orphans") or []
                             if o.get("action") == "remove"),
        "erased_words": sum(_int(v) for k, v in erased.items() if k != "erased_facts"),
        "unfinished_updates": sum(_int(v) for k, v in obligations.items()
                                  if k != "needs_review"),
        "memories_without_fact": _int((plan.get("memories_without_own_fact") or {})
                                      .get("to_repair")),
    }


def read_state(root: Path) -> dict[str, Any]:
    state = read_json(Path(root) / STATE_FILE) or {}
    return {**state, "running": _busy.locked()}


def _save(root: Path, **fields: Any) -> dict[str, Any]:
    state = {**(read_json(Path(root) / STATE_FILE) or {}), **fields}
    write_json_atomic(Path(root) / STATE_FILE, state)
    return state


def _check(db_path: Path, root: Path, version: str) -> dict[str, Any]:
    from superlocalmemory.storage.integrity_scan import plan

    started = time.monotonic()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        found = findings(plan(conn, foreign_keys=False))
    finally:
        conn.close()
    logger.info("memory store checked in %.0f s: %d item(s) to repair",
                time.monotonic() - started, sum(found.values()))
    return _save(root, status="checked", checked_at=_now(), checked_version=version,
                 findings=found, to_repair=sum(found.values()))


def _in_background(name: str, work: Any) -> bool:
    """Run ``work`` in one daemon thread unless a check or repair is running."""
    if not _busy.acquire(blocking=False):
        return False

    def run() -> None:
        try:
            work()
        except Exception as exc:  # never into the daemon
            logger.warning("memory store %s did not finish: %s", name, exc)
        finally:
            _busy.release()

    threading.Thread(target=run, daemon=True, name=f"store-{name}").start()
    return True


def start_check(db_path: Path, root: Path, version: str, *, delay_s: float = 0.0) -> bool:
    """Check the store in the background (read-only). False when one is running."""
    def work() -> None:
        if delay_s:
            time.sleep(delay_s)
        try:
            _check(Path(db_path), Path(root), version)
        except Exception as exc:
            _save(root, status="check_failed", checked_at=_now(), error=str(exc)[:300])
            raise
    return _in_background("check", work)


def check_after_upgrade(db_path: Path, root: Path, version: str) -> bool:
    """Once per installed version: the first start after an upgrade checks."""
    if (read_json(Path(root) / STATE_FILE) or {}).get("checked_version") == version:
        return False
    return start_check(db_path, root, version, delay_s=FIRST_CHECK_DELAY_S)


def start_repair(db_path: Path, root: Path, version: str, *, engine: Any = None) -> bool:
    """Back up, repair, check again; in the background. False when busy."""
    def work() -> None:
        from superlocalmemory.infra.backup import BackupManager
        from superlocalmemory.storage.integrity_repair import Repair

        _save(root, status="repairing", repair_started_at=_now(), error=None)
        copy = BackupManager(db_path=Path(db_path), base_dir=Path(root)).create_backup(
            label=BACKUP_LABEL)
        if not copy:
            _save(root, status="repair_failed", error="The backup copy could not be "
                  "made, so nothing was changed.")
            return
        try:
            summary = Repair(Path(db_path), engine=engine).apply()
        except Exception as exc:
            _save(root, status="repair_failed", backup=copy, error=str(exc)[:300])
            raise
        _save(root, repaired_at=_now(), backup=copy,
              repair_status=str(summary.get("status")), repair_run=summary.get("run_id"))
        _check(Path(db_path), Path(root), version)
        _save(root, status="repaired")
    return _in_background("repair", work)


__all__ = ["BACKUP_LABEL", "FINDINGS", "FIRST_CHECK_DELAY_S", "STATE_FILE",
           "check_after_upgrade", "findings", "read_state", "start_check", "start_repair"]
