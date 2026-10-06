# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A version number in a question finds the memory about that version.

The keyword index splits ``4.1.20`` into the numbers ``4``, ``1`` and ``20``.
On a real store those numbers are everywhere (times, counts, dates), so each
one carries almost no weight and every 4.1.x memory scored the same. These
tests build that situation on purpose: the filler memories below make 4, 1,
20 and "release" common, exactly as they are on a live store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.retrieval.bm25_channel import BM25Channel
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.fts_terms import version_match_phrases, version_terms
from superlocalmemory.storage.models import AtomicFact, MemoryRecord

_VERSION_FACTS = {
    "f-4.1.0": "4.1.0 added working memory",
    "f-4.1.20": "4.1.20 added answer check",
    "f-4.1.2": "4.1.2 fixed M043",
}


def _filler(i: int) -> str:
    # Common numbers and a common word, no dotted versions.
    return f"Release train {i}: room 4 booked for item {i % 7}, slot 1:20"


def _store(db: DatabaseManager, fact_id: str, content: str) -> None:
    db.store_memory(MemoryRecord(memory_id=f"m-{fact_id}", content=content))
    db.store_fact(AtomicFact(
        fact_id=fact_id, memory_id=f"m-{fact_id}", content=content,
    ))


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    for fid, content in _VERSION_FACTS.items():
        _store(mgr, fid, content)
    for i in range(40):
        _store(mgr, f"filler-{i:02d}", _filler(i))
    return mgr


#: A short memory holding all three numbers, but no version. Before the fix
#: it won the near-tie every 4.1.x memory was stuck in.
_NUMBERS_ONLY = "Lunch 1:20 room 4"

#: How far the right memory must lead on the channel's [0, 1) scale. Before
#: the fix every candidate scored ~0.000001, so any real margin proves the
#: version itself is now evidence and the order is no longer an accident.
_MATERIAL_LEAD = 0.1


def _ranked(db: DatabaseManager, query: str) -> list[str]:
    return [fid for fid, _ in BM25Channel(db).search(query, "default", top_k=30)]


class TestVersionQueriesRankTheRightRelease:
    def test_what_is_in_a_release_ranks_that_release_first(self, db) -> None:
        _store(db, "numbers-only", _NUMBERS_ONLY)
        scored = BM25Channel(db).search("what's in 4.1.20", "default", top_k=30)
        assert scored[0][0] == "f-4.1.20"
        assert scored[0][1] - scored[1][1] > _MATERIAL_LEAD, scored[:3]

    def test_a_common_word_does_not_outrank_the_exact_version(self, db) -> None:
        _store(db, "f-4.1.0-release", "The 4.1.0 release added working memory")
        assert _ranked(db, "what's in the 4.1.20 release")[0] == "f-4.1.20"

    def test_a_shorter_version_is_not_confused_with_a_longer_one(self, db) -> None:
        # 4.1.2 must not be found inside 4.1.20, nor the other way round.
        assert _ranked(db, "what did 4.1.2 fix")[0] == "f-4.1.2"

    def test_a_leading_v_matches_in_either_direction(self, db) -> None:
        _store(db, "f-v4.1.20", "shipped v4.1.20 to everyone")
        for query in ("what's in 4.1.20", "what's in v4.1.20", "V4.1.20"):
            assert set(_ranked(db, query)[:2]) == {"f-4.1.20", "f-v4.1.20"}, query

    def test_a_longer_version_containing_it_does_not_win(self, db) -> None:
        # "4.1.20.1" contains the words 4, 1, 20 in a row, but it is a
        # different version. The shorter memory is the exact one.
        _store(db, "f-4.1.20.1", "4.1.20.1 hotfix")
        _store(db, "f-3.4.1.20", "3.4.1.20 build")
        ranked = _ranked(db, "4.1.20")
        assert ranked[0] == "f-4.1.20"
        assert ranked.index("f-4.1.20.1") > 0
        assert ranked.index("f-3.4.1.20") > 0

    def test_a_partial_version_still_finds_its_releases(self, db) -> None:
        ranked = _ranked(db, "4.1")
        assert set(_VERSION_FACTS) <= set(ranked)

    def test_the_same_question_ranks_the_same_way_twice(self, db) -> None:
        first = BM25Channel(db).search("what's in 4.1.20", "default", top_k=30)
        second = BM25Channel(db).search("what's in 4.1.20", "default", top_k=30)
        assert first == second

    def test_a_question_without_a_version_is_unchanged(self, db) -> None:
        assert _ranked(db, "answer check")[0] == "f-4.1.20"


class TestTheOtherKeywordPathsAgree:
    def test_full_text_search_ranks_the_exact_release_first(self, db) -> None:
        _store(db, "numbers-only", _NUMBERS_ONLY)
        facts = db.search_facts_fts("what's in 4.1.20", "default", limit=10)
        assert facts[0].fact_id == "f-4.1.20"

    def test_the_index_free_fallback_ranks_the_exact_release_first(
        self, db,
    ) -> None:
        # A store without the FTS5 table takes the in-memory path.
        for trigger in ("insert", "delete", "update"):
            db.execute(f"DROP TRIGGER IF EXISTS atomic_facts_fts_{trigger}")
        db.execute("DROP TABLE atomic_facts_fts")
        _store(db, "numbers-only", _NUMBERS_ONLY)
        assert _ranked(db, "what's in 4.1.20")[0] == "f-4.1.20"
        assert _ranked(db, "what did 4.1.2 fix")[0] == "f-4.1.2"


class TestVersionTerms:
    @pytest.mark.parametrize(("text", "expected"), [
        ("what's in 4.1.20", ("4.1.20",)),
        ("v4.1.20 and V4.1.2", ("4.1.20", "4.1.2")),
        ("shipped 4.1.20.", ("4.1.20",)),
        ("4.1.20.1 hotfix", ("4.1.20.1",)),
        ("4.1.20 then 4.1.20 again", ("4.1.20",)),
        ("4.1.20rc1", ()),
        ("dev4.1.20", ()),
        ("no version here, just 20 and 4", ()),
        ("192.168.1.1", ("192.168.1.1",)),
    ])
    def test_finds_whole_dotted_numbers(self, text, expected) -> None:
        assert version_terms(text) == expected

    def test_phrases_cover_the_leading_v_spelling(self) -> None:
        assert version_match_phrases("is 4.1.20 out") == (
            '"4.1.20"', '"v4.1.20"',
        )
