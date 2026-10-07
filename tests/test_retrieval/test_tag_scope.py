# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What a recall's ``tags`` filter did, in words (4.1.22 G05)."""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.facets import Facets
from superlocalmemory.retrieval.tag_scope import build_report
from superlocalmemory.retrieval.tag_search import tag_members
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _save(db, content, *, profile="default", tags=None) -> str:
    if profile != "default":
        db.execute(
            "INSERT OR IGNORE INTO profiles(profile_id, name) VALUES (?, ?)",
            (profile, profile),
        )
    meta = {"tags": tags} if tags is not None else {}
    memory_id = db.store_memory(MemoryRecord(profile_id=profile, content=content, metadata=meta))
    return db.store_fact(AtomicFact(profile_id=profile, memory_id=memory_id, content=content,
                                    fact_type=FactType.SEMANTIC))


# -- tag_members --------------------------------------------------------

def test_tag_members_finds_the_right_ids(db) -> None:
    tagged = _save(db, "x", tags="decision")
    other = _save(db, "y", tags="status")
    members = tag_members(db, "default", ("decision",), "all")
    assert members == {tagged}
    assert other not in members


def test_tag_members_personal_scope_only(db) -> None:
    """4.1.22 CRIT: cross-profile leakage. The membership scan is
    ``WHERE f.profile_id = ?`` — another profile's identically tagged memory
    must never appear, even though both stores share one sqlite file."""
    _save(db, "mine", profile="default", tags="shared-label")
    other = _save(db, "theirs", profile="other", tags="shared-label")
    members = tag_members(db, "default", ("shared-label",), "all")
    assert other not in members
    other_members = tag_members(db, "other", ("shared-label",), "all")
    assert other in other_members


def test_tag_members_empty_wanted_is_empty(db) -> None:
    _save(db, "x", tags="decision")
    assert tag_members(db, "default", (), "all") == frozenset()
    assert tag_members(db, "default", ("   ",), "all") == frozenset()


def test_tag_members_any_vs_all(db) -> None:
    both = _save(db, "x", tags="a,b")
    one = _save(db, "y", tags="a")
    assert tag_members(db, "default", ("a", "b"), "all") == {both}
    assert tag_members(db, "default", ("a", "b"), "any") == {both, one}


# -- build_report ---------------------------------------------------------

def test_report_when_matched_is_nonzero_never_reads_the_store() -> None:
    # No real db at all: if the matched>0 short-circuit ever regressed into
    # reading the store, this would raise instead of returning cleanly.
    report = build_report(None, "default", Facets.of(tags="decision"), matched=3)
    assert report == {
        "tags": ["decision"], "keys": ["decision"], "match": "all",
        "applied": True, "matched": 3, "note": "",
    }
    assert "reason" not in report


def test_report_no_memory_has_tag(db) -> None:
    _save(db, "x", tags="unrelated")
    report = build_report(db, "default", Facets.of(tags="decision"), matched=0)
    assert report["applied"] is True
    assert report["matched"] == 0
    assert report["reason"] == "no_memory_has_tag"
    assert "decision" in report["note"]


def test_report_none_relevant(db) -> None:
    # Something IS tagged 'decision' in this profile, just not among the
    # (zero) candidates this particular recall admitted.
    _save(db, "x", tags="decision")
    report = build_report(db, "default", Facets.of(tags="decision"), matched=0)
    assert report["reason"] == "none_relevant"


def test_report_distinguishes_profiles(db) -> None:
    """A tag that exists only in ANOTHER profile must still read as
    'no_memory_has_tag' for this one — the existence check is personal-scope,
    same reach as the filter and the search-inside supplement."""
    _save(db, "theirs", profile="other", tags="decision")
    report = build_report(db, "default", Facets.of(tags="decision"), matched=0)
    assert report["reason"] == "no_memory_has_tag"


def test_report_any_match_semantics(db) -> None:
    _save(db, "x", tags="a")
    report = build_report(db, "default", Facets.of(tags=["a", "b"], tags_match="any"), matched=0)
    # 'a' alone satisfies "any", so something DOES exist under these semantics.
    assert report["reason"] == "none_relevant"


def test_report_all_match_semantics_needs_every_label(db) -> None:
    _save(db, "x", tags="a")
    report = build_report(db, "default", Facets.of(tags=["a", "b"], tags_match="all"), matched=0)
    # No single memory has BOTH a and b, so under "all" nothing exists.
    assert report["reason"] == "no_memory_has_tag"


def test_report_says_unreadable_when_the_membership_read_fails(db, monkeypatch) -> None:
    """tag_members swallows a DB error and returns None; the report must call
    that unreadable, never "no memory has the tag" (a claim nobody checked)."""
    _save(db, "x", tags="decision")

    def broken(*_a, **_k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "execute", broken)
    report = build_report(db, "default", Facets.of(tags="decision"), matched=0)
    assert report["reason"] == "unreadable"
    assert "could not be checked" in report["note"]
