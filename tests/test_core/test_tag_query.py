# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Exact tag filtering for ``search`` and ``list_recent`` (4.1.22 G05).

The tagged memory's own words never contain the tag label, so only the
filter can find it, and it is always placed behind a crowd that an ordinary
fetch would return first. Synthetic text throughout.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import tag_query
from superlocalmemory.core.kind_query import list_recent_facts, search_facts
from superlocalmemory.core.tag_query import TagFilter
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

CROWD = 300


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _save(db, content, *, tags=None, kind=None, profile="default") -> str:
    if profile != "default":
        db.execute("INSERT OR IGNORE INTO profiles(profile_id, name) VALUES (?, ?)",
                   (profile, profile))
    meta = {"tags": tags} if tags is not None else {}
    memory_id = db.store_memory(MemoryRecord(profile_id=profile, content=content,
                                             metadata=meta))
    return db.store_fact(AtomicFact(
        profile_id=profile, memory_id=memory_id, content=content,
        fact_type=FactType.SEMANTIC, memory_kind=kind,
        memory_kind_source="user" if kind else None))


def _ids(facts) -> list[str]:
    return [f.fact_id for f in facts]


def test_list_finds_an_old_tagged_memory_behind_many_newer_ones(db) -> None:
    tagged = _save(db, "The team chose a manual approval gate.", tags="decision-record")
    for i in range(CROWD):
        _save(db, f"Synthetic status note {i}.", tags="status")
    assert tagged not in _ids(list_recent_facts(db, "default", 5, None))
    found = list_recent_facts(db, "default", 5, None,
                              tag_filter=TagFilter.of("decision-record"))
    assert _ids(found) == [tagged]


def test_search_finds_the_tagged_match_behind_closer_untagged_matches(db) -> None:
    tagged = _save(db, "Kestrel checkpoint uses a manual approval gate.",
                   tags="decision-record")
    for i in range(CROWD):
        _save(db, f"Kestrel checkpoint kestrel checkpoint status {i}.")
    assert tagged not in _ids(search_facts(db, "kestrel checkpoint", "default", 5, None))
    found = search_facts(db, "kestrel checkpoint", "default", 5, None,
                         tag_filter=TagFilter.of(["decision-record"]))
    assert _ids(found) == [tagged]


def test_search_with_tags_still_needs_the_words(db) -> None:
    _save(db, "Kestrel checkpoint uses a manual approval gate.", tags="decision-record")
    assert search_facts(db, "unrelated words", "default", 5, None,
                        tag_filter=TagFilter.of("decision-record")) == []


def test_kind_filters_inside_the_tag_set(db) -> None:
    rule = _save(db, "Always tag releases.", tags="release", kind="rule")
    _save(db, "We picked the blue build.", tags="release", kind="decision")
    _save(db, "Never skip review.", tags="other", kind="rule")
    for i in range(CROWD):
        _save(db, f"Untagged rule {i}.", kind="rule")
    found = list_recent_facts(db, "default", 5, "rule", tag_filter=TagFilter.of("release"))
    assert _ids(found) == [rule]


def test_all_and_any(db) -> None:
    both = _save(db, "one", tags="alpha,beta")
    only_a = _save(db, "two", tags="alpha")
    _save(db, "three", tags="gamma")
    all_ = list_recent_facts(db, "default", 10, None,
                             tag_filter=TagFilter.of(["alpha", "beta"]))
    any_ = list_recent_facts(db, "default", 10, None,
                             tag_filter=TagFilter.of(["alpha", "beta"], "any"))
    assert set(_ids(all_)) == {both}
    assert set(_ids(any_)) == {both, only_a}


def test_case_spacing_and_stored_shape_do_not_matter(db) -> None:
    csv = _save(db, "csv form", tags="Token-Optimization, perf")
    listed = _save(db, "list form", tags=["token-optimization"])
    found = list_recent_facts(db, "default", 10, None,
                              tag_filter=TagFilter.of(" token-OPTIMIZATION "))
    assert set(_ids(found)) == {csv, listed}


def test_another_profiles_tagged_memory_is_never_listed(db) -> None:
    mine = _save(db, "mine", tags="shared-label")
    _save(db, "theirs", tags="shared-label", profile="other")
    found = list_recent_facts(db, "default", 10, None,
                              tag_filter=TagFilter.of("shared-label"))
    assert _ids(found) == [mine]


def test_without_tags_nothing_changes(db) -> None:
    for i in range(20):
        _save(db, f"note {i} kestrel", tags="x" if i % 2 else None)
    assert _ids(list_recent_facts(db, "default", 7, None, tag_filter=TagFilter.of(None))) \
        == _ids(list_recent_facts(db, "default", 7, None))
    assert _ids(search_facts(db, "kestrel", "default", 7, None,
                             tag_filter=TagFilter.of(""))) \
        == _ids(search_facts(db, "kestrel", "default", 7, None))


def test_an_unreadable_membership_keeps_nothing_and_says_so(db, monkeypatch) -> None:
    _save(db, "tagged", tags="decision-record")
    monkeypatch.setattr("superlocalmemory.retrieval.tag_search.tag_members",
                        lambda *a, **k: None)
    tf = TagFilter.of("decision-record")
    assert list_recent_facts(db, "default", 10, None, tag_filter=tf) == []
    out = tag_query.with_report({"count": 0}, db, "default", tf)
    assert out["tag_scope"]["reason"] == "unreadable"


def test_report_tells_an_unknown_tag_from_an_unmatched_one(db) -> None:
    _save(db, "tagged words here", tags="decision-record")
    tf = TagFilter.of("decision-record")
    assert search_facts(db, "absent", "default", 5, None, tag_filter=tf) == []
    assert tag_query.with_report({"count": 0}, db, "default", tf)["tag_scope"]["reason"] \
        == "none_relevant"
    nobody = TagFilter.of("never-used")
    assert tag_query.with_report({"count": 0}, db, "default", nobody)["tag_scope"]["reason"] \
        == "no_memory_has_tag"
    assert "tag_scope" not in tag_query.with_report({"count": 0}, db, "default",
                                                    TagFilter.of(None))


def test_a_deleted_tagged_memory_is_not_listed(db) -> None:
    gone = _save(db, "to be removed", tags="decision-record")
    kept = _save(db, "kept", tags="decision-record")
    db.delete_fact(gone)
    found = list_recent_facts(db, "default", 10, None,
                              tag_filter=TagFilter.of("decision-record"))
    assert _ids(found) == [kept]
