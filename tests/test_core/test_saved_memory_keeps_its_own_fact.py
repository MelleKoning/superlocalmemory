# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""A saved memory stays findable after enrichment, even when it looks like an
older memory.

Found by the release check: two notes that differ only in a number ("the relay
uses channel 4" / "channel 9") or in their declared kind were judged
near-duplicates during enrichment, and the newer note's own first-queryable
fact was DELETED in favour of the older memory's fact. The newer memory was
left with no fact at all: findable right after saving, unfindable a second
later, its number and its declared kind gone. A copy of a live store had 35
such memories, 33 of them carrying numbers.

A near-duplicate of ANOTHER memory no longer removes this memory's own
fact. A near-duplicate inside the same save is still folded away.
"""

from __future__ import annotations

from unittest.mock import patch

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingestion_command import IngestionRequest, IngestionState
from superlocalmemory.storage.migrations import M018_ingestion_operations
from superlocalmemory.storage.models import ConsolidationAction, ConsolidationActionType


def _save(engine, command, key: str, content: str):
    receipt = command.submit(IngestionRequest(
        content=content, profile_id=engine._profile_id, source_type="http",
        idempotency_key=key, trusted_actor_id="daemon-capability:owned-instance",
        metadata={"kind": "rule"}))
    assert receipt.state is IngestionState.QUERYABLE
    return receipt


def test_a_near_duplicate_of_another_memory_keeps_this_memory_s_own_fact(
        engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    command = build_engine_ingestion_command(engine)

    first = _save(engine, command, "k1", "The harbor relay uses channel 4.")
    assert command.materialize(first.operation_id).state is IngestionState.COMPLETE
    older_fact = first.queryable_fact_ids[0]

    second = _save(engine, command, "k2", "The harbor relay uses channel 9.")
    own_fact = second.queryable_fact_ids[0]
    memory_id = engine._db.get_fact(own_fact).memory_id

    def near_duplicate_of_the_older_memory(fact, profile_id, **_kw):
        return ConsolidationAction(profile_id=profile_id,
                                   action_type=ConsolidationActionType.NOOP,
                                   new_fact_id=fact.fact_id, existing_fact_id=older_fact,
                                   reason="near-duplicate (score=0.961)")

    with patch.object(engine._fact_extractor, "extract_facts", return_value=[]), \
            patch.object(engine._consolidator, "consolidate",
                         side_effect=near_duplicate_of_the_older_memory):
        done = command.materialize(second.operation_id)

    assert done.state is IngestionState.COMPLETE
    kept = engine._db.get_facts_by_memory_id(memory_id, engine._profile_id)
    assert [f.fact_id for f in kept] == [own_fact], \
        "the newer memory was left without its own fact"
    assert "channel 9" in kept[0].content
    assert engine._db.get_fact(older_fact) is not None
