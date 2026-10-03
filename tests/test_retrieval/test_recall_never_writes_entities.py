# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A recall reads the store; it never changes it and never waits on a model.

Asking about a name SLM has never seen used to create a permanent, empty
entity for it, save aliases, rewrite last-seen times, and - with a local
model configured - stop the recall for several seconds while the model
guessed what the name meant. Saving a memory still does all of that; a
question does none of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.encoding.entity_resolver import (
    JARO_WINKLER_AUTO_MERGE,
    JARO_WINKLER_LLM_FLOOR,
    EntityResolver,
    jaro_winkler,
)
from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import CanonicalEntity, EntityAlias

PROFILE = "default"
CLOSE_SPELLING = "Alice Smitth"   # above the automatic-match bar
UNSURE_SPELLING = "Alyse Smit"    # in the band that used to go to the model


class _ModelThatMustNotBeAsked:
    is_available = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        raise AssertionError("a recall asked the language model")


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    mgr.store_entity(CanonicalEntity(
        entity_id="e_alice", profile_id=PROFILE, canonical_name="Alice Smith",
    ))
    mgr.store_alias(EntityAlias(
        alias_id="a_ali", entity_id="e_alice", alias="Ali", source="test",
    ), PROFILE)
    return mgr


@pytest.fixture()
def model() -> _ModelThatMustNotBeAsked:
    return _ModelThatMustNotBeAsked()


@pytest.fixture()
def resolver(db: DatabaseManager, model: _ModelThatMustNotBeAsked) -> EntityResolver:
    return EntityResolver(db=db, llm=model)


def _store_state(db: DatabaseManager) -> tuple:
    entities = tuple(tuple(dict(r).values()) for r in db.execute(
        "SELECT * FROM canonical_entities ORDER BY entity_id"))
    aliases = tuple(tuple(dict(r).values()) for r in db.execute(
        "SELECT * FROM entity_aliases ORDER BY alias_id"))
    return entities, aliases


def test_the_spellings_used_here_sit_where_the_test_needs_them() -> None:
    assert jaro_winkler(CLOSE_SPELLING.lower(), "alice smith") >= JARO_WINKLER_AUTO_MERGE
    unsure = jaro_winkler(UNSURE_SPELLING.lower(), "alice smith")
    assert JARO_WINKLER_LLM_FLOOR <= unsure < JARO_WINKLER_AUTO_MERGE


def test_a_question_finds_known_names_by_name_alias_and_close_spelling(
    resolver: EntityResolver,
) -> None:
    found = resolver.lookup(["Alice Smith", "Ali", CLOSE_SPELLING], PROFILE)
    assert found == {"Alice Smith": "e_alice", "Ali": "e_alice", CLOSE_SPELLING: "e_alice"}


def test_a_question_leaves_unknown_and_uncertain_names_unmatched(
    resolver: EntityResolver,
) -> None:
    assert resolver.lookup(["Zorblax", UNSURE_SPELLING], PROFILE) == {}


def test_a_question_changes_nothing_and_asks_no_model(
    resolver: EntityResolver, db: DatabaseManager, model: _ModelThatMustNotBeAsked,
) -> None:
    before = _store_state(db)
    resolver.lookup(["Zorblax", UNSURE_SPELLING, CLOSE_SPELLING, "Ali"], PROFILE)
    assert _store_state(db) == before
    assert model.calls == 0


def test_the_recall_channel_reads_without_writing(
    resolver: EntityResolver, db: DatabaseManager, model: _ModelThatMustNotBeAsked,
) -> None:
    channel = EntityGraphChannel(db, resolver)
    before = _store_state(db)
    ids = channel._resolve_entities(["Zorblax", UNSURE_SPELLING, "Alice Smith"], PROFILE)
    assert ids == ["e_alice"]
    assert _store_state(db) == before
    assert model.calls == 0


def test_saving_a_memory_still_records_new_names(db: DatabaseManager) -> None:
    store_side = EntityResolver(db=db, llm=None)
    before = _store_state(db)
    created = store_side.resolve(["Zorblax"], PROFILE)
    assert "Zorblax" in created
    assert _store_state(db) != before
