# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""``slm db repair`` gives back the searchable fact that enrichment wrongly
removed from a memory (before 4.1.22 a near-duplicate of ANOTHER memory lost
its only fact), and never brings back anything erased, deleted or changed."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingestion_command import IngestionRequest, IngestionState
from superlocalmemory.storage import own_fact_repair as own
from superlocalmemory.storage.integrity_repair import Limits, Repair
from superlocalmemory.storage.migrations import M018_ingestion_operations
from superlocalmemory.storage.models import ConsolidationAction, ConsolidationActionType


@pytest.fixture
def engine(engine_with_mock_deps):
    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    return engine_with_mock_deps


def _broken(engine, key: str, words: str, *, older: str) -> tuple[str, str]:
    """A memory whose own fact the pre-4.1.22 rule removed: (memory_id, fact_id)."""
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=words, profile_id=engine._profile_id, source_type="http",
        idempotency_key=key, trusted_actor_id="daemon-capability:owned-instance",
        metadata={"_slm_memory_kind": "rule", "tags": "harbor"}))
    own_fact = receipt.queryable_fact_ids[0]
    memory_id = engine._db.get_fact(own_fact).memory_id

    def noop(fact, profile_id, **_kw):
        return ConsolidationAction(profile_id=profile_id, action_type=ConsolidationActionType.NOOP,
                                   new_fact_id=fact.fact_id, existing_fact_id=older,
                                   reason="near-duplicate (score=0.961)")
    with patch.object(engine._fact_extractor, "extract_facts", return_value=[]), \
            patch.object(engine._consolidator, "consolidate", side_effect=noop), \
            patch("superlocalmemory.core.store_pipeline.keeps_own_fact", return_value=False):
        assert command.materialize(receipt.operation_id).state is IngestionState.COMPLETE
    engine._db.execute("INSERT INTO consolidation_log (action_id, profile_id, action_type, "
                       "new_fact_id, existing_fact_id, reason, timestamp) VALUES "
                       "(?, ?, 'noop', ?, ?, 'near-duplicate', ?)",
                       (f"log-{key}", engine._profile_id, own_fact, older, "2026-10-01"))
    assert engine._db.get_facts_by_memory_id(memory_id, engine._profile_id) == []
    return memory_id, own_fact


def _older(engine) -> str:
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content="The harbor relay uses channel 4.", profile_id=engine._profile_id,
        source_type="http", idempotency_key="older",
        trusted_actor_id="daemon-capability:owned-instance"))
    command.materialize(receipt.operation_id)
    return receipt.queryable_fact_ids[0]


def _repair(engine) -> dict:
    return Repair(engine._db.db_path, limits=Limits(pause_s=0, confirm_s=0)).apply()


def _facts(engine, memory_id):
    return engine._db.get_facts_by_memory_id(memory_id, engine._profile_id)


def test_the_memory_gets_its_own_fact_back_and_is_found_by_its_words(engine) -> None:
    older = _older(engine)
    memory_id, _ = _broken(engine, "k9", "The harbor relay uses channel 9.", older=older)
    with engine._db.raw_connection() as conn:
        assert own.census(conn)["to_repair"] == 1

    summary = _repair(engine)

    assert summary["done"].get("own_facts.restored") == 1
    [fact] = _facts(engine, memory_id)
    assert "channel 9" in fact.content
    assert fact.memory_kind == "rule" and fact.memory_kind_source == "caller"
    # Queued for enrichment like a save, then found by its own words.
    [op] = [dict(r) for r in engine._db.execute(
        "SELECT operation_id, state, queryable_fact_ids_json FROM ingestion_operations "
        "WHERE idempotency_key = ?", (f"own-fact-repair:{memory_id}",))]
    assert op["state"] == "queryable" and json.loads(op["queryable_fact_ids_json"]) == [fact.fact_id]
    done = build_engine_ingestion_command(engine).materialize(op["operation_id"])
    assert done.state is IngestionState.COMPLETE
    assert [f.fact_id for f in _facts(engine, memory_id)] == [fact.fact_id]
    found = {r.fact.memory_id for r in engine.recall("harbor relay channel 9", limit=10).results}
    assert memory_id in found
    assert _repair(engine)["done"].get("own_facts.restored") is None  # idempotent

    Repair(engine._db.db_path).undo(summary["run_id"])
    found = {r.fact.memory_id for r in engine.recall("harbor relay channel 9", limit=10).results}
    assert memory_id not in found, "undo left the restored fact findable"


def _exclusion(engine, how: str) -> None:
    db = engine._db
    older = _older(engine)
    memory_id, fact_id = _broken(engine, how, f"The harbor relay uses channel {len(how)}.",
                                 older=older)
    op = json.loads(dict(db.execute("SELECT metadata_json FROM memories WHERE memory_id=?",
                                    (memory_id,))[0])["metadata_json"])["ingestion_operation_id"]
    if how == "tombstone":
        db.execute("INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, memory_id, "
                   "created_at) VALUES (?, ?, 'e1', ?, ?)",
                   (engine._profile_id, fact_id, memory_id, time.time()))
    elif how in ("erasure_receipt", "profile_erased", "entity_erased_later"):
        subject_type, subject = {"erasure_receipt": ("fact", fact_id),
                                 "profile_erased": ("profile", engine._profile_id),
                                 "entity_erased_later": ("entity", "ent-1")}[how]
        db.execute("INSERT INTO erasure_receipts (erasure_id, profile_id, subject_type, "
                   "subject_id, state, audit_hash, requested_at, completed_at) "
                   "VALUES (?, ?, ?, ?, 'COMPLETE', 'h', ?, ?)",
                   (f"er-{how}", engine._profile_id, subject_type, subject,
                    time.time(), time.time()))
    elif how == "erasing_now":
        from superlocalmemory.storage.erasure_fence import mark_erasing
        mark_erasing(engine._profile_id, fact_id)
    elif how == "scrubbed":
        db.execute("UPDATE ingestion_operations SET raw_content='', raw_metadata_json='{}' "
                   "WHERE operation_id=?", (op,))
    elif how == "unfinished":
        db.execute("UPDATE ingestion_operations SET state='enriching' WHERE operation_id=?", (op,))
    elif how == "other_profile":
        db.execute("UPDATE ingestion_operations SET profile_id='someone-else' "
                   "WHERE operation_id=?", (op,))
    elif how == "empty_text":
        db.execute("UPDATE memories SET content='' WHERE memory_id=?", (memory_id,))
    elif how == "withheld_fact":  # its fact is quarantined (withheld), not missing
        db.execute("UPDATE atomic_facts SET memory_id=?, quarantined=1 WHERE fact_id=?",
                   (memory_id, older))
    elif how == "deleted_by_person":  # its own fact was deleted: no proof enrichment did it
        db.execute("DELETE FROM consolidation_log WHERE new_fact_id=?", (fact_id,))
        db.execute("UPDATE ingestion_operations SET final_fact_ids_json=? WHERE operation_id=?",
                   (json.dumps([fact_id]), op))
    try:
        _repair(engine)
    finally:
        if how == "erasing_now":
            from superlocalmemory.storage.erasure_fence import clear_erasing
            clear_erasing(engine._profile_id, fact_id)
    visible = [f for f in _facts(engine, memory_id) if f.fact_id != older]
    assert visible == [], f"{how}: the memory came back"


@pytest.mark.parametrize("how", [
    "tombstone", "erasure_receipt", "profile_erased", "entity_erased_later", "erasing_now",
    "scrubbed", "unfinished", "other_profile", "deleted_by_person", "empty_text",
    "withheld_fact"])
def test_what_must_never_come_back(engine, how) -> None:
    _exclusion(engine, how)


def test_a_save_folded_into_other_memories_is_repaired_without_a_log_row(engine) -> None:
    """Older versions left no near-duplicate log row; the save's own record
    still proves it: every fact it finished with belongs to another memory."""
    older = _older(engine)
    memory_id, fact_id = _broken(engine, "k7", "The harbor relay uses channel 7.", older=older)
    engine._db.execute("DELETE FROM consolidation_log WHERE new_fact_id=?", (fact_id,))
    with engine._db.raw_connection() as conn:
        assert own.census(conn)["to_repair"] == 1
    _repair(engine)
    assert [f.content for f in _facts(engine, memory_id)] == ["The harbor relay uses channel 7."]
