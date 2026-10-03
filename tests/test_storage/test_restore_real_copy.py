# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Upgrade -> snapshot -> verify -> restore on a COPY of a real user store.

Skipped unless ``SLM_REAL_SNAPSHOT`` names a folder holding a ``memory.db`` (and
optionally ``learning.db``) taken from a real installation. The files are
copied into a scratch folder first (``SLM_REAL_WORKDIR`` or pytest's tmp_path);
the source is only ever read. Run with ``-s`` to see the timings.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import upgrade_restore as ur

REAL = os.environ.get("SLM_REAL_SNAPSHOT")


def _facts_by_type(db: Path) -> dict[str, int]:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        return dict(conn.execute(
            "SELECT fact_type, COUNT(*) FROM atomic_facts GROUP BY fact_type").fetchall())


def _checks(db: Path) -> tuple[str, bool]:
    with closing(sqlite3.connect(str(db))) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        try:
            conn.execute("INSERT INTO atomic_facts_fts(atomic_facts_fts) VALUES('integrity-check')")
            fts = True
        except sqlite3.DatabaseError:
            fts = False
    return integrity, fts


@pytest.mark.skipif(not REAL, reason="set SLM_REAL_SNAPSHOT to a folder with a real memory.db")
def test_real_store_round_trip(tmp_path) -> None:
    source = Path(REAL)
    work = Path(os.environ.get("SLM_REAL_WORKDIR") or tmp_path) / f"run-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    for name in ("memory.db", "learning.db"):
        if (source / name).exists():
            shutil.copyfile(source / name, work / name)   # a quiescent backup, not a live store
    memory_db, learning_db = work / "memory.db", work / "learning.db"
    before_types = _facts_by_type(memory_db)
    timings: dict[str, float] = {}

    t = time.monotonic()
    result = mr.apply_all(learning_db, memory_db)            # the 4.1.19 upgrade
    timings["upgrade (snapshot+manifest+M052)"] = time.monotonic() - t
    assert "M052_memory_kinds" in result["applied"], result["details"]
    assert _facts_by_type(memory_db) == before_types
    [point] = ur.list_restore_points(work)
    t = time.monotonic()
    ur.verify_point(point)
    timings["verify (sha256 + quick_check)"] = time.monotonic() - t

    with closing(sqlite3.connect(memory_db)) as conn:       # life after the upgrade
        conn.execute("PRAGMA foreign_keys=ON")
        victim = conn.execute(
            "SELECT fact_id FROM atomic_facts f WHERE NOT EXISTS (SELECT 1 FROM "
            "correction_cases c WHERE c.predecessor_fact_id=f.fact_id OR "
            "c.successor_fact_id=f.fact_id) LIMIT 1").fetchone()[0]
        conn.execute("DELETE FROM atomic_facts WHERE fact_id=?", (victim,))
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES "
                     "('after-upgrade', 'default', 'Written after the upgrade')")
        conn.commit()

    t = time.monotonic()
    preview = ur.preview_restore(point.point_id, data_root=work, memory_db=memory_db)
    timings["preview"] = time.monotonic() - t
    assert preview.verified and (preview.deleted_facts, preview.added_memories) == (1, 1)
    ur.request_restore(point.point_id, requested_by="real-copy", data_root=work,
                       memory_db=memory_db)
    t = time.monotonic()
    outcome = ur.perform_pending_restore(work, memory_db, learning_db)
    timings["restore (delta+staging+safety copy+write)"] = time.monotonic() - t

    assert outcome.status == "restored", outcome.message
    after_types = _facts_by_type(memory_db)
    assert sum(after_types.values()) == sum(before_types.values()) - 1
    assert _checks(memory_db) == ("ok", True)
    t = time.monotonic()
    again = mr.apply_all(learning_db, memory_db)              # the next start
    timings["re-apply M052 on restored store"] = time.monotonic() - t
    assert "M052_memory_kinds" in again["applied"]
    assert _checks(memory_db) == ("ok", True)
    print("\nREAL COPY:", {"facts_before": sum(before_types.values()),
                           "facts_after": sum(after_types.values()),
                           "snapshot_bytes": point.size_bytes,
                           **{k: round(v, 2) for k, v in timings.items()}})
