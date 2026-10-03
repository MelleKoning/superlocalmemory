# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Kinds a person confirmed after the copy survive a restore; the newer edit wins (L1-09).

A. A fact the copy already had, re-confirmed after the copy: the restored row
   carries the copy's older confirmation, which must not block the newer one.
B. A memory added after the copy, with one of its facts confirmed: the memory is
   re-ingested (new fact ids), so the confirmation is matched to the fact with
   the same words once that memory has finished enriching.
C. A confirmation made after the restore is newer than anything in the export.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from superlocalmemory.storage import _restore_reimport as ri
from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._durable_json import read_json

from ._upgrade_store import add_memory, confirm_kind, current_store

_FUTURE = "2099-01-01T00:00:00+00:00"


def _kind(db: Path, fact_id: str) -> tuple:
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute("SELECT memory_kind, memory_kind_source FROM atomic_facts "
                            "WHERE fact_id=?", (fact_id,)).fetchone()


def _engine(db: Path):
    return SimpleNamespace(_db=SimpleNamespace(db_path=db))


def _scenario(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["We ship on Fridays"], fact_ids=["f1"])
    confirm_kind(memory_db, "f1", "decision")
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    confirm_kind(memory_db, "f1", "rule")                                     # A
    add_memory(memory_db, "m9", ["Run tests before merge", "Coverage stays above 80"],
               fact_ids=["g1", "g2"])
    confirm_kind(memory_db, "g1", "rule")                                     # B
    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)
    assert outcome.status == "restored" and outcome.reimport_pending
    return memory_db, preview, Path(outcome.delta_dir)


def _reingested(memory_db: Path, state: str = "complete") -> None:
    """What the canonical re-import of m9 leaves once its enrichment ran."""
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES "
                     "('new-m9', 'default', 'Run tests before merge Coverage stays above 80')")
        for fid, text in (("n1", "Run tests before merge"), ("n2", "Coverage stays above 80")):
            conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                         "VALUES (?, 'new-m9', 'default', ?)", (fid, text))
        conn.execute(
            "INSERT INTO ingestion_operations (operation_id, profile_id, source_type, "
            "idempotency_key, source_hash, raw_content, state, final_fact_ids_json) "
            "VALUES (?, 'default', 'restore-reimport', 'restore-reimport:m9', 'h', 'x', ?, ?)",
            (uuid.uuid4().hex, state, json.dumps(["n1", "n2"] if state == "complete" else [])))
        conn.commit()


def _set_state(memory_db: Path, state: str) -> None:
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("UPDATE ingestion_operations SET state=?, final_fact_ids_json=? "
                     "WHERE idempotency_key='restore-reimport:m9'",
                     (state, json.dumps(["n1", "n2"])))
        conn.commit()


def _no_memories(monkeypatch) -> None:
    def already(_engine, rows, _db, counts):
        counts["already_present"] += len(rows)
    monkeypatch.setattr(ri, "_reimport_memories", already)


def test_a_newer_confirmation_of_a_copied_fact_wins(tmp_path, monkeypatch) -> None:
    memory_db, preview, delta = _scenario(tmp_path)
    _reingested(memory_db)
    _no_memories(monkeypatch)
    assert _kind(memory_db, "f1") == ("decision", "user"), "the copy's older choice"

    report = ur.reimport_delta(_engine(memory_db), delta, data_root=tmp_path)

    assert _kind(memory_db, "f1") == ("rule", "user")
    assert preview.kind_edits == 2 and report.kinds_reapplied == 2, report


def test_a_confirmation_on_a_memory_added_after_the_copy_survives(tmp_path, monkeypatch) -> None:
    memory_db, _preview, delta = _scenario(tmp_path)
    _reingested(memory_db)
    _no_memories(monkeypatch)

    ur.reimport_delta(_engine(memory_db), delta, data_root=tmp_path)

    assert _kind(memory_db, "n1") == ("rule", "user")
    assert _kind(memory_db, "n2") == (None, None), "only the fact that was confirmed"


def test_it_waits_for_the_memory_to_finish_enriching(tmp_path, monkeypatch) -> None:
    memory_db, _preview, delta = _scenario(tmp_path)
    _reingested(memory_db, state="queryable")
    _no_memories(monkeypatch)

    first = ur.reimport_delta(_engine(memory_db), delta, data_root=tmp_path)

    assert first.kinds_waiting == 1 and _kind(memory_db, "n1") == (None, None)
    assert read_json(tmp_path / "restore-outcome.json")["reimport_pending"] is True
    assert delta.is_dir(), "kept while a confirmation is waiting"
    _set_state(memory_db, "complete")
    second = ur.run_pending_reimport(_engine(memory_db), tmp_path)
    assert second.kinds_waiting == 0 and _kind(memory_db, "n1") == ("rule", "user")
    assert read_json(tmp_path / "restore-outcome.json")["reimport_pending"] is False


def test_a_confirmation_made_after_the_restore_is_newer(tmp_path, monkeypatch) -> None:
    memory_db, _preview, delta = _scenario(tmp_path)
    _reingested(memory_db)
    _no_memories(monkeypatch)
    with closing(sqlite3.connect(memory_db)) as conn:       # the person re-chose, later
        conn.execute("UPDATE atomic_facts SET memory_kind='preference', "
                     "memory_kind_source='user', memory_kind_at=? WHERE fact_id IN ('f1','n1')",
                     (_FUTURE,))
        conn.commit()

    ur.reimport_delta(_engine(memory_db), delta, data_root=tmp_path)

    assert _kind(memory_db, "f1") == ("preference", "user")
    assert _kind(memory_db, "n1") == ("preference", "user")
