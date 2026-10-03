# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Changing a memory's kind goes through the canonical writer: confined to the
owner's profile, recorded per fact, idempotent, and kept by corrections."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
    M033_projection_transactions,
    M034_obligation_integrity,
    M042_correction_case_ledger,
)
from superlocalmemory.storage.migrations import M052_memory_kinds as m052
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture()
def env(tmp_path):
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    for migration in (M018_ingestion_operations, M032_write_coordinator_admission,
                      M033_projection_transactions, M034_obligation_integrity):
        migration.apply(conn)
    conn.commit()
    conn.close()
    db = DatabaseManager(path)
    db.initialize(schema)
    with db.raw_connection() as raw:
        M042_correction_case_ledger.apply(raw)
        m052.apply(raw)
    db.execute("INSERT OR IGNORE INTO profiles(profile_id, name, description) "
               "VALUES ('other', 'other', 'test')")
    runtime = CanonicalRememberRuntime(db=db, profile_id="default",
                                       writer=lambda _r, _o: [],
                                       journal_path=tmp_path / "journal.db")
    runtime.start()
    yield db, runtime
    runtime.stop()


def _fact(db: DatabaseManager, profile_id: str, content: str) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id=profile_id, content=content))
    return db.store_fact(AtomicFact(profile_id=profile_id, memory_id=memory_id,
                                    content=content, fact_type=FactType.SEMANTIC))


def _kind(db, fact_id):
    return dict(db.execute("SELECT memory_kind, memory_kind_source, fact_type "
                           "FROM atomic_facts WHERE fact_id=?", (fact_id,))[0])


def test_owner_profile_only(env) -> None:
    db, runtime = env
    mine = _fact(db, "default", "Always run the suite once before release.")
    theirs = _fact(db, "other", "Their own private rule.")
    receipt = runtime.set_fact_kinds("default", [(mine, "rule"), (theirs, "rule")],
                                     idempotency_key="kinds-1")
    assert receipt["ok"] is True and receipt["changed"] == 1
    assert _kind(db, mine) == {"memory_kind": "rule", "memory_kind_source": "user",
                               "fact_type": "semantic"}
    assert _kind(db, theirs)["memory_kind"] is None


def test_history_row_per_changed_fact(env) -> None:
    db, runtime = env
    ids = [_fact(db, "default", f"Decision number {i} for the release.") for i in range(3)]
    runtime.set_fact_kinds("default", [(i, "decision") for i in ids], idempotency_key="k2")
    rows = db.execute("SELECT fact_id, origin, new_kind, actor FROM memory_kind_history")
    assert sorted(dict(r)["fact_id"] for r in rows) == sorted(ids)
    assert {(dict(r)["origin"], dict(r)["new_kind"]) for r in rows} == {("user_edit", "decision")}


def test_more_than_200_items_rejected(env) -> None:
    _db, runtime = env
    with pytest.raises(ValueError):
        runtime.set_fact_kinds("default", [(f"f{i}", "rule") for i in range(201)])
    with pytest.raises(ValueError):
        runtime.set_fact_kinds("default", [])


def test_an_unknown_kind_rejects_the_whole_request(env) -> None:
    db, runtime = env
    fid = _fact(db, "default", "Keep the 3 second recall ceiling.")
    with pytest.raises(ValueError):
        runtime.set_fact_kinds("default", [(fid, "rule"), (fid, "not-a-kind")])
    assert _kind(db, fid)["memory_kind"] is None


def test_same_idempotency_key_replays_receipt(env) -> None:
    db, runtime = env
    fid = _fact(db, "default", "Ship the answer check on by default.")
    first = runtime.set_fact_kinds("default", [(fid, "decision")], idempotency_key="same")
    second = runtime.set_fact_kinds("default", [(fid, "decision")], idempotency_key="same")
    assert first == second
    assert len(db.execute("SELECT 1 FROM memory_kind_history WHERE fact_id=?", (fid,))) == 1


def test_correction_successor_inherits_kind(env) -> None:
    db, runtime = env
    fid = _fact(db, "default", "Recall ceiling is 2 seconds.")
    runtime.set_fact_kinds("default", [(fid, "decision")], idempotency_key="k-corr")
    runtime.create_correction_successor("default", fid, "succ-1",
                                        "Recall ceiling is 3 seconds.",
                                        idempotency_key="corr-1")
    assert _kind(db, "succ-1")["memory_kind"] == "decision"
    assert _kind(db, "succ-1")["memory_kind_source"] == "user"
