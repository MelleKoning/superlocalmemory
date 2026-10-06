# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A kind declared on a save whose words are already stored is never dropped silently.

Runs the real queryable writer (the one every remember door reaches) on a
real store with the kind schema: an unconfirmed stored fact takes the caller's
kind, auditable; a different confirmed kind is kept and reported.
"""

from __future__ import annotations

import uuid

import pytest

from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.core.kind_on_resave import kind_conflict
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.memory_kinds import METADATA_KEY
from superlocalmemory.storage.migrations import M018_ingestion_operations, M052_memory_kinds

WORDS = "Fixture record KR-1: the shared synthetic code is K991."
ACTOR = "daemon-capability:" + "9" * 64


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(tmp_path / "memory.db")
    manager.initialize(schema)
    with manager.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M052_memory_kinds.apply(conn)
    manager._kind_columns_present = True  # noqa: SLF001 - the migration ran above
    return manager


def _save(db, kind: str | None = None, *, source_type: str = "http") -> list[str]:
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=build_immediate_admission_handler(db, profile_id="default"),
        materialize=lambda *_: [],
    )
    metadata = {METADATA_KEY: kind} if kind else {}
    receipt = command.submit(IngestionRequest(
        content=WORDS, profile_id="default", source_type=source_type,
        idempotency_key=uuid.uuid4().hex, metadata=metadata, trusted_actor_id=ACTOR,
    ))
    return list(receipt.fact_ids)


def _kind(db, fact_id: str) -> tuple:
    row = db.execute("SELECT memory_kind, memory_kind_source, fact_type FROM atomic_facts "
                     "WHERE fact_id=?", (fact_id,))[0]
    return row["memory_kind"], row["memory_kind_source"], row["fact_type"]


def _history(db, fact_id: str) -> list[dict]:
    return [dict(r) for r in db.execute(
        "SELECT origin, old_kind, old_source, new_kind, new_source, actor "
        "FROM memory_kind_history WHERE fact_id=? ORDER BY history_id", (fact_id,))]


def test_a_declared_kind_confirms_identical_words_stored_without_one(db) -> None:
    first = _save(db)
    assert _kind(db, first[0])[1] != "caller"

    second = _save(db, "rule")

    assert second == first  # still one fact for identical words
    assert _kind(db, first[0]) == ("rule", "caller", "semantic")
    assert _history(db, first[0]) == [{
        "origin": "user_edit", "old_kind": None, "old_source": None,
        "new_kind": "rule", "new_source": "caller", "actor": ACTOR,
    }]
    assert kind_conflict(db, fact_ids=second, profile_id="default", declared="rule") is None


def test_a_machine_suggestion_is_replaced_by_the_declaration(db) -> None:
    first = _save(db)
    db.execute("UPDATE atomic_facts SET memory_kind='status', memory_kind_source='rules' "
               "WHERE fact_id=?", (first[0],))

    _save(db, "decision")

    assert _kind(db, first[0])[:2] == ("decision", "caller")
    assert _history(db, first[0])[0]["old_source"] == "rules"


def test_a_different_confirmed_kind_is_kept_and_reported(db) -> None:
    first = _save(db, "status")

    second = _save(db, "opinion")

    assert second == first
    assert _kind(db, first[0])[:2] == ("status", "caller")
    assert _history(db, first[0]) == []
    assert kind_conflict(db, fact_ids=second, profile_id="default", declared="opinion") == {
        "kept": "status", "requested": "opinion"}


def test_the_same_kind_again_changes_nothing(db) -> None:
    first = _save(db, "procedure")

    _save(db, "procedure")

    assert _kind(db, first[0])[:2] == ("procedure", "caller")
    assert _history(db, first[0]) == []
    assert kind_conflict(db, fact_ids=first, profile_id="default", declared="procedure") is None


def test_an_automatic_capture_cannot_confirm_a_kind(db) -> None:
    # Hooks and observers may not confirm a standing rule (KIND_TRUSTED_SOURCES).
    first = _save(db)

    _save(db, "rule", source_type="hook")

    assert _kind(db, first[0])[1] != "caller"
    assert _history(db, first[0]) == []


def test_no_conflict_is_reported_without_a_declaration(db) -> None:
    first = _save(db, "status")
    assert kind_conflict(db, fact_ids=first, profile_id="default", declared=None) is None
    assert kind_conflict(db, fact_ids=[], profile_id="default", declared="rule") is None
