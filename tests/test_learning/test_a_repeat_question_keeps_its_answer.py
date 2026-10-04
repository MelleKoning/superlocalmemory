# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Asking the same question again does not change its answer (R7).

The recent-winner cap stops a memory compounding the outcome bonus once it has
won first place three times. It counted every recall, so asking ONE question a
fourth time capped that question's own top answer and flipped it — the answer
changed because the person asked again. The cap exists to stop one memory
dominating MANY questions, so it now counts wins across different questions.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.recall_pipeline import _apply_outcome_bonus
from superlocalmemory.learning.pcos import RECENT_TOPS, RecentTopCounter
from superlocalmemory.storage.models import AtomicFact, RetrievalResult

P = "r7-profile"


@pytest.fixture(autouse=True)
def _fresh_counter():
    RECENT_TOPS.forget(P)
    yield
    RECENT_TOPS.forget(P)


@pytest.fixture()
def memory_db(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE fact_outcome_score (fact_id TEXT, profile_id TEXT,"
                 " score REAL, play_count INTEGER, PRIMARY KEY(fact_id, profile_id))")
    # B has helped repeatedly: its bonus lifts it over A's retrieval lead.
    conn.execute("INSERT INTO fact_outcome_score VALUES ('B', ?, 0.9, 20)", (P,))
    conn.commit()
    conn.close()
    return db


def _ask(db, query):
    results = [RetrievalResult(fact=AtomicFact(fact_id="A", content="A"), score=0.50),
               RetrievalResult(fact=AtomicFact(fact_id="B", content="B"), score=0.45)]
    return [r.fact.fact_id for r in _apply_outcome_bonus(results, P, db, query=query)]


def test_the_same_question_six_times_keeps_one_answer(memory_db) -> None:
    orders = {tuple(_ask(memory_db, "which db do we use?")) for _ in range(6)}
    assert orders == {("B", "A")}


def test_a_retyped_question_is_the_same_question(memory_db) -> None:
    for q in ("which db do we use?", "Which DB do we use?", "  which db  do we use? "):
        assert _ask(memory_db, q) == ["B", "A"]
    assert _ask(memory_db, "which db do we use?") == ["B", "A"]


def test_the_cap_still_bites_across_different_questions(memory_db) -> None:
    for q in ("question one", "question two", "question three"):
        assert _ask(memory_db, q) == ["B", "A"]
    # B has now won three DIFFERENT questions: it stops earning the bonus.
    assert _ask(memory_db, "question four") == ["A", "B"]


class TestTheCounter:
    def test_one_question_counts_once(self) -> None:
        c = RecentTopCounter()
        for _ in range(5):
            c.record_top("p", "f", query_key="q")
        assert c.tops("p", "f") == 1 and not c.capped("p", "f")

    def test_distinct_questions_count(self) -> None:
        c = RecentTopCounter()
        for q in ("a", "b", "c"):
            c.record_top("p", "f", query_key=q)
        assert c.capped("p", "f")

    def test_decay_keeps_the_most_recent_questions(self) -> None:
        c = RecentTopCounter()
        for q in ("a", "b", "c", "d"):
            c.record_top("p", "hot", query_key=q)
        for i in range(RecentTopCounter._WINDOW + 1):
            c.record_top("p", f"other-{i}", query_key=f"x{i}")
        assert c.tops("p", "hot") == 2
