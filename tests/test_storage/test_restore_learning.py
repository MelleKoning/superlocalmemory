# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""learning.db across a restore (L1-05).

An erased profile stays erased in learning.db as well as in the memory store,
and the preview says how much learning recorded since the copy is not carried
over (the choice is to warn, not merge: see ``_restore_learning``).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import add_memory, add_profile, current_store, fact_ids, record_erasure


def _learning_rows(learning_db: Path, profile: str) -> tuple[int, int]:
    with closing(sqlite3.connect(learning_db)) as conn:
        return (conn.execute("SELECT COUNT(*) FROM bandit_plays WHERE profile_id=?",
                             (profile,)).fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM learning_feedback WHERE profile_id=?",
                             (profile,)).fetchone()[0])


def _play(learning_db: Path, profile: str, query: str, fact_id: str) -> None:
    with closing(sqlite3.connect(learning_db)) as conn:
        conn.execute("INSERT INTO bandit_plays (profile_id, query_id, stratum, arm_id, "
                     "played_at, shown_fact_ids) VALUES (?, ?, 's', 'arm', '2026-10-01', ?)",
                     (profile, query, f'["{fact_id}"]'))
        conn.execute("INSERT INTO learning_feedback (profile_id, fact_id, signal_type, "
                     "signal_value, created_at) VALUES (?, ?, 'click', 1.0, '2026-10-01')",
                     (profile, fact_id))
        conn.commit()


def test_an_erased_profile_stays_erased_in_learning(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_profile(memory_db, "alice")
    ids = add_memory(memory_db, "a1", ["Alice's private medical note"], profile="alice")
    _play(learning_db, "alice", "q1", ids[0])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    with closing(sqlite3.connect(memory_db)) as conn:       # Alice is erased after the copy
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM atomic_facts WHERE profile_id='alice'")
        conn.execute("DELETE FROM memories WHERE profile_id='alice'")
        conn.commit()
    record_erasure(memory_db, profile_id="alice", subject_type="profile", subject_id="alice",
                   fact_ids=ids, memory_ids={ids[0]: "a1"})
    with closing(sqlite3.connect(learning_db)) as conn:
        conn.execute("DELETE FROM bandit_plays WHERE profile_id='alice'")
        conn.execute("DELETE FROM learning_feedback WHERE profile_id='alice'")
        conn.commit()

    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "restored", outcome.message
    assert fact_ids(memory_db, "alice") == set()
    assert _learning_rows(learning_db, "alice") == (0, 0)
    assert outcome.applied["learning_rows_erased"] == 2


def test_preview_states_the_learning_that_is_not_carried_over(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    ids = add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"])
    _play(learning_db, "default", "q1", ids[0])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    _play(learning_db, "default", "q2", ids[0])              # learned after the copy
    _play(learning_db, "default", "q3", ids[0])

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)

    assert preview.learning_records_lost == 4               # 2 plays + 2 feedback rows
    assert any("4 learning records" in w and "not carried over" in w
               for w in preview.warnings), preview.warnings


def test_no_learning_warning_when_nothing_was_learned(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)

    assert preview.learning_records_lost == 0
    assert not any("learning" in w for w in preview.warnings)
