# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Shared direct-engine kind filtering for ``list_recent``/``list`` and
``search`` (LLD/WP8 4.1.19) — one implementation, called by both the MCP
tools and the CLI, so the two surfaces cannot disagree.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.kind_query import (
    InvalidKind,
    list_recent_facts,
    resolve_kind,
    search_facts,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _save(db, content, *, kind=None, source=None, fact_type=FactType.SEMANTIC) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(
        profile_id="default", memory_id=memory_id, content=content, fact_type=fact_type,
        memory_kind=kind, memory_kind_source=source,
    )
    return db.store_fact(fact)


# -- resolve_kind -------------------------------------------------------------


def test_resolve_kind_empty_is_no_filter() -> None:
    assert resolve_kind("") is None
    assert resolve_kind("   ") is None
    assert resolve_kind(None) is None


def test_resolve_kind_normalizes_a_known_alias() -> None:
    assert resolve_kind("todo") == "prospective"
    assert resolve_kind(" Decision ") == "decision"


def test_resolve_kind_rejects_an_unknown_word() -> None:
    with pytest.raises(InvalidKind):
        resolve_kind("not-a-real-kind")


# -- list_recent_facts ---------------------------------------------------------


def test_list_recent_with_no_kind_returns_newest_first(db) -> None:
    _save(db, "a")
    _save(db, "b")
    facts = list_recent_facts(db, "default", 10, None)
    assert [f.content for f in facts] == ["b", "a"]


def test_list_recent_kind_filter_keeps_only_matches(db) -> None:
    _save(db, "rule one", kind="rule", source="user")
    _save(db, "a decision", kind="decision", source="user")
    _save(db, "rule two", kind="rule", source="user")
    facts = list_recent_facts(db, "default", 10, "rule")
    assert {f.content for f in facts} == {"rule one", "rule two"}


def test_list_recent_kind_filter_matches_legacy_mapped_kind(db) -> None:
    # No memory_kind of its own; kind_fields() maps episodic fact_type -> "episodic".
    _save(db, "something happened", fact_type=FactType.EPISODIC)
    _save(db, "a fact", fact_type=FactType.SEMANTIC)
    facts = list_recent_facts(db, "default", 10, "episodic")
    assert [f.content for f in facts] == ["something happened"]


def test_list_recent_kind_filter_overfetches_to_still_fill_the_limit(db) -> None:
    # 5 non-matching rows ahead of 2 matching ones in recency order: with no
    # overfetch a limit=2 fetch would see only the first 2 (both "status") and
    # return nothing for kind="rule".
    for i in range(5):
        _save(db, f"status {i}", kind="status", source="user")
    _save(db, "rule a", kind="rule", source="user")
    _save(db, "rule b", kind="rule", source="user")
    facts = list_recent_facts(db, "default", 2, "rule")
    assert {f.content for f in facts} == {"rule a", "rule b"}


# -- search_facts --------------------------------------------------------------


def test_search_with_no_kind_is_unfiltered(db) -> None:
    _save(db, "alpha fact")
    _save(db, "alpha decision", kind="decision", source="user")
    facts = search_facts(db, "alpha", "default", 10, None)
    assert {f.content for f in facts} == {"alpha fact", "alpha decision"}


def test_search_kind_filter_keeps_only_matches(db) -> None:
    _save(db, "alpha fact")
    _save(db, "alpha decision", kind="decision", source="user")
    facts = search_facts(db, "alpha", "default", 10, "decision")
    assert [f.content for f in facts] == ["alpha decision"]
