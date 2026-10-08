# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Own-fact repair never publishes erased or withheld text, and stays correct
under races, failures and undo (independent audit findings A1-A5, CRIT-3)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from superlocalmemory.core import kind_assignment
from superlocalmemory.storage import own_fact_repair as own
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.integrity_repair import Limits, Repair
from superlocalmemory.storage.migrations import M018_ingestion_operations
from tests.test_storage.test_own_fact_repair import _broken, _older


@pytest.fixture
def engine(engine_with_mock_deps):
    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    return engine_with_mock_deps


def _live(engine, memory_id: str) -> list[str]:
    return [dict(r)["fact_id"] for r in engine._db.execute(
        "SELECT fact_id FROM atomic_facts WHERE memory_id = ? AND archive_status = 'live' "
        "AND quarantined = 0", (memory_id,))]


def _repair(engine) -> dict:
    return Repair(engine._db.db_path, limits=Limits(pause_s=0, confirm_s=0)).apply()


def _erase_fact(engine, fact_id: str, memory_id: str | None) -> None:
    now = time.time()
    db = engine._db
    db.execute("INSERT INTO erasure_receipts (erasure_id, profile_id, subject_type, subject_id, "
               "state, audit_hash, requested_at, completed_at) VALUES (?, ?, 'fact', ?, "
               "'COMPLETE', 'h', ?, ?)", (f"er-{fact_id}", engine._profile_id, fact_id, now, now))
    db.execute("INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, memory_id, "
               "created_at) VALUES (?, ?, ?, ?, ?)",
               (engine._profile_id, fact_id, f"er-{fact_id}", memory_id, now))


def test_a1_an_erasure_between_listing_and_repair_is_honoured(engine, monkeypatch) -> None:
    older = _older(engine)
    memory_id, fact_id = _broken(engine, "a1", "The harbor relay uses channel 9.", older=older)
    with engine._db.raw_connection() as conn:
        stale = own.classify(conn)              # listed while still repairable
    assert [c.memory_id for c in stale[0]] == [memory_id]
    _erase_fact(engine, fact_id, memory_id)     # erasure completes before the write
    monkeypatch.setattr(own, "classify", lambda _conn: stale)
    _repair(engine)
    assert _live(engine, memory_id) == [], "erased text was published again"


def test_a2_a_fact_erasure_of_the_sibling_it_was_folded_into_holds_it_back(engine) -> None:
    older = _older(engine)
    memory_id, _ = _broken(engine, "a2", "The harbor relay uses channel 9.", older=older)
    older_memory = engine._db.get_fact(older).memory_id
    _erase_fact(engine, older, older_memory)
    engine._db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (older,))
    with engine._db.raw_connection() as conn:
        found, held = own.classify(conn)
    assert memory_id not in [c.memory_id for c in found], "near-identical erased text comes back"
    assert held["erased"] == 1


def test_a3_a_withheld_sibling_holds_it_back(engine) -> None:
    older = _older(engine)
    memory_id, _ = _broken(engine, "a3", "The harbor relay uses channel 9.", older=older)
    engine._db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (older,))
    with engine._db.raw_connection() as conn:
        found, held = own.classify(conn)
    assert memory_id not in [c.memory_id for c in found], "withheld text would be published"
    assert held["withheld"] == 1


def test_a4_two_writers_create_one_fact(engine, monkeypatch) -> None:
    older = _older(engine)
    memory_id, _ = _broken(engine, "a4", "The harbor relay uses channel 9.", older=older)
    with engine._db.raw_connection() as conn:
        [candidate], _ = own.classify(conn)
    real = kind_assignment.store_fact_keeping_kind
    gate = threading.Barrier(2, timeout=3)

    def both_checked_then_store(*a, **k):
        try:
            gate.wait()          # both writers have checked "no fact yet"
        except threading.BrokenBarrierError:
            pass                 # the other writer is (correctly) kept out
        return real(*a, **k)
    monkeypatch.setattr(kind_assignment, "store_fact_keeping_kind", both_checked_then_store)
    results, errors = [], []

    def writer():
        # Another process: its own connection and its own process-local write
        # lock, so only the database itself can keep the two apart.
        try:
            db = DatabaseManager(engine._db.db_path)
            db._lock = threading.RLock()
            conn = sqlite3.connect(str(engine._db.db_path), timeout=30)
            try:
                results.append(own.promote(db, conn, candidate))
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=writer) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert not errors, errors
    assert len(_live(engine, memory_id)) == 1, "two writers both created the fact"
    assert sorted(r is None for r in results) == [False, True]


def test_a5_one_failing_memory_does_not_stop_the_others(engine, monkeypatch) -> None:
    older = _older(engine)
    first, _ = _broken(engine, "a5a", "The harbor relay uses channel 9.", older=older)
    second, _ = _broken(engine, "a5b", "The harbor relay uses channel 8.", older=older)
    real = kind_assignment.store_fact_keeping_kind
    calls = []

    def fail_first(db, fact, **k):
        calls.append(fact.memory_id)
        if fact.memory_id == first:
            raise RuntimeError("disk said no")
        return real(db, fact, **k)
    monkeypatch.setattr(kind_assignment, "store_fact_keeping_kind", fail_first)
    try:
        summary = _repair(engine)
    except RuntimeError:
        summary = None
    assert summary is not None, "one failing memory aborted the repair"
    assert len(_live(engine, second)) == 1 and _live(engine, first) == []
    assert summary["done"].get("own_facts.failed") == 1
    rows = engine._db.execute("SELECT after_json FROM integrity_repair_receipts "
                              "WHERE action = 'own_fact_failed'")
    assert rows and "RuntimeError" in json.loads(dict(rows[0])["after_json"])["error"]


def test_crit3_undo_leaves_nothing_to_enrich_and_nothing_findable(engine) -> None:
    from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command

    older = _older(engine)
    memory_id, _ = _broken(engine, "c3", "The harbor relay uses channel 9.", older=older)
    summary = _repair(engine)
    Repair(engine._db.db_path).undo(summary["run_id"])
    [op] = [dict(r) for r in engine._db.execute(
        "SELECT operation_id, state FROM ingestion_operations WHERE idempotency_key = ?",
        (f"own-fact-repair:{memory_id}",))]
    due = {o.operation_id for o in
           build_engine_ingestion_command(engine).repository.list_materializable(limit=50)}
    assert op["operation_id"] not in due, "undo left the repair queued for enrichment"
    assert _live(engine, memory_id) == []


@pytest.mark.parametrize("enriched_first", [True])
def test_crit3_undo_after_enrichment_hides_every_fact_of_the_memory(engine, enriched_first) -> None:
    from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
    from superlocalmemory.storage.models import AtomicFact, FactType

    older = _older(engine)
    memory_id, _ = _broken(engine, "c3b", "The harbor relay uses channel 9.", older=older)
    summary = _repair(engine)
    # enrichment may have derived more facts for the memory before the undo
    engine._db.store_fact(AtomicFact(fact_id="derived-c3b", memory_id=memory_id,
                                     profile_id=engine._profile_id, content="relay channel 9",
                                     fact_type=FactType.SEMANTIC))
    assert build_engine_ingestion_command(engine) is not None
    Repair(engine._db.db_path).undo(summary["run_id"])
    assert _live(engine, memory_id) == [], "a derived fact stayed findable after undo"


def test_an_unrelated_fact_erasure_in_the_profile_does_not_hold_it_back(engine) -> None:
    """Re-creating the fact exposes only this memory's own stored text; erasing an
    unrelated fact elsewhere in the profile says nothing about it."""
    older = _older(engine)
    memory_id, _ = _broken(engine, "un", "The harbor relay uses channel 9.", older=older)
    _erase_fact(engine, "unrelated-fact", "unrelated-memory")
    with engine._db.raw_connection() as conn:
        found, held = own.classify(conn)
    assert [c.memory_id for c in found] == [memory_id], held
    _repair(engine)
    assert len(_live(engine, memory_id)) == 1
