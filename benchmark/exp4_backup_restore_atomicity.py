"""Experiment 4 — Backup-restore atomicity (rollback on partial failure).

Guarantee: if a restore fails partway through, the live data set is rolled back
to exactly the content that was live before the restore began — never a mix of
old and new epochs, never the half-written backup epoch.

Method: for each trial we build a real ``BackupCoordinator`` over two live
SQLite stores, capture a backup set, replace the live stores with a distinct
"new epoch", then inject a disk error on the second store's write, AFTER the
first store has been written. The real restore must raise
``BackupRestoreError``, roll the first store back, and leave both live stores
at the new-epoch content (the pre-restore snapshot), with no snapshot residue.
Content, not bytes: the restore writes through SQLite, which rewrites header
counters. Fault injection wraps the coordinator's ``_write_into_live_store`` on
the instance; the coordinator's code is unmodified.

Until 4.1.18 the injection patched ``shutil.copy2`` and failed its second call,
which was the Phase-B snapshot of the second store — so every trial aborted
before any live file was written, and no rollback was ever exercised.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from unittest.mock import patch

from _harness import TempWorkspace, TrialOutcome, run_trials


def _make_sqlite(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS marker (val TEXT)")
    conn.execute("INSERT INTO marker VALUES (?)", (marker,))
    conn.commit()
    conn.close()


def _markers(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return [r[0] for r in conn.execute("SELECT val FROM marker ORDER BY rowid")]
    finally:
        conn.close()


def _trial(index: int) -> TrialOutcome:
    from superlocalmemory.infra.backup import BackupCoordinator, BackupRestoreError

    subset = ("memory.db", "learning.db")
    with TempWorkspace() as ws:
        base_dir = ws / "live"
        backup_dir = ws / "backups"
        base_dir.mkdir()
        backup_dir.mkdir()

        tag = uuid.uuid4().hex[:6]
        _make_sqlite(base_dir / "memory.db", f"orig_mem_{tag}")
        _make_sqlite(base_dir / "learning.db", f"orig_learn_{tag}")

        coord = BackupCoordinator(
            managed_databases=subset, base_dir=base_dir, backup_dir=backup_dir,
        )
        manifest = coord.create_backup_set()

        # Replace live stores with a distinguishable new epoch.
        _make_sqlite(base_dir / "memory.db", f"new_mem_{tag}")
        _make_sqlite(base_dir / "learning.db", f"new_learn_{tag}")
        new_mem = _markers(base_dir / "memory.db")
        new_learn = _markers(base_dir / "learning.db")

        # Inject a failure on the SECOND store's write; the first is real, as
        # are the rollback writes that follow.
        real_write = coord._write_into_live_store
        calls: list[str] = []

        def _failing_write(source, target):
            calls.append(target.name)
            if len(calls) == 2:
                raise OSError("injected disk error on second store")
            return real_write(source, target)

        raised = False
        with patch.object(coord, "_write_into_live_store", _failing_write):
            try:
                coord.restore_from_manifest(manifest)
            except BackupRestoreError:
                raised = True

        partial_write = calls[:2] == ["memory.db", "learning.db"]
        rolled_back = (
            _markers(base_dir / "memory.db") == new_mem
            and _markers(base_dir / "learning.db") == new_learn
        )
        no_residue = (
            not list(base_dir.glob("*.pre_restore"))
            and not list(base_dir.glob("*.restore_staging"))
        )
        held = raised and partial_write and rolled_back and no_residue
        detail = {"index": index}
        if not held:
            detail.update(raised=raised, partial_write=partial_write,
                          rolled_back=rolled_back, no_residue=no_residue)
        return TrialOutcome(index=index, held=held, detail=detail)


def run(n_trials: int = 200, seed: int = 0):
    return run_trials(
        name="exp4_backup_restore_atomicity",
        guarantee="partial-restore failure rolls live data back to pre-restore content",
        metric_name="rollback+clean rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real BackupCoordinator create/restore; failure injected on the "
            "second store's write after the first store was written; asserts "
            "BackupRestoreError, pre-restore content intact, no "
            "staging/snapshot residue."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} "
          f"({result.metric_value:.4f})")
