# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A bandit arm is credited only for an order it chose.

With the opt-in hosted reorder on, the order a person sees first is the hosted
model's, not the arm's. Rewarding the arm for how that order went teaches it
noise. The play is still settled — and the memories the outcome named are
still credited, because the person did see them — but the arm's posterior is
left alone.

And the play's record of what was shown is compared against the list the play
actually recorded, not against a list taken after later passes moved things.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from superlocalmemory.core import recall_pipeline
from superlocalmemory.learning.bandit import ORDER_REPLACED, ContextualBandit
from superlocalmemory.learning.reward_from_outcomes import settle_from_outcomes
from superlocalmemory.retrieval.jev_rerank import STATUS_LISTWISE
from superlocalmemory.storage.migrations import M005_bandit_tables as _M005
from superlocalmemory.storage.migrations import M044_play_carries_its_own_evidence as _M044
from superlocalmemory.storage.migrations import M045_fact_outcome_score as _M045

_PROFILE = "default"


@pytest.fixture()
def store(tmp_path: Path) -> tuple[Path, Path]:
    learning = tmp_path / "learning.db"
    conn = sqlite3.connect(str(learning))
    conn.executescript(_M005.DDL)
    _M044.apply(conn)
    conn.commit()
    conn.close()
    memory = tmp_path / "memory.db"
    conn = sqlite3.connect(str(memory))
    conn.execute(
        "CREATE TABLE action_outcomes ("
        " outcome_id TEXT PRIMARY KEY, profile_id TEXT, query TEXT,"
        " fact_ids_json TEXT, outcome TEXT, context_json TEXT,"
        " timestamp TEXT, reward REAL, settled INTEGER, settled_at TEXT,"
        " recall_query_id TEXT)"
    )
    _M045.apply(conn)
    conn.commit()
    conn.close()
    return learning, memory


def _play(learning: Path, shown: list[str]) -> int:
    bandit = ContextualBandit(learning, profile_id=_PROFILE)
    choice = bandit.choose({"query_type": "open_domain", "entity_count": 0}, "q-1")
    assert choice.play_id
    assert bandit.record_shown(choice.play_id, shown)
    return choice.play_id


def _row(learning: Path, play_id: int) -> sqlite3.Row:
    conn = sqlite3.connect(str(learning))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM bandit_plays WHERE play_id = ?",
                            (play_id,)).fetchone()
    finally:
        conn.close()


def _arms(learning: Path) -> list[tuple]:
    conn = sqlite3.connect(str(learning))
    try:
        return conn.execute("SELECT arm_id, alpha, beta, plays FROM bandit_arms").fetchall()
    finally:
        conn.close()


def _report(memory: Path, facts: list[str], reward: float, at: datetime) -> None:
    conn = sqlite3.connect(str(memory))
    conn.execute(
        "INSERT INTO action_outcomes (outcome_id, profile_id, query, fact_ids_json,"
        " outcome, context_json, timestamp, reward, settled, settled_at, recall_query_id)"
        " VALUES (?,?,'',?,'success','{}',?,?,1,?,'')",
        (f"o-{at.timestamp()}", _PROFILE, json.dumps(facts), at.isoformat(), reward,
         at.isoformat()),
    )
    conn.commit()
    conn.close()


def _fact_score(memory: Path, fact_id: str):
    conn = sqlite3.connect(str(memory))
    try:
        return conn.execute("SELECT * FROM fact_outcome_score WHERE fact_id = ?",
                            (fact_id,)).fetchone()
    finally:
        conn.close()


class TestTheArmIsNotCreditedForAReplacedOrder:
    def test_a_marked_play_settles_without_moving_the_arm(self, store) -> None:
        learning, memory = store
        play_id = _play(learning, ["aaaa", "bbbb"])
        assert ContextualBandit(learning, _PROFILE).mark_order_replaced(play_id)

        at = datetime.fromisoformat(_row(learning, play_id)["played_at"]) + timedelta(seconds=90)
        _report(memory, ["aaaa"], 1.0, at)
        assert settle_from_outcomes(_PROFILE, learning, memory,
                                    now=at + timedelta(seconds=30)) == 1

        row = _row(learning, play_id)
        assert row["settled_at"] is not None, "the play must still be closed"
        assert row["reward"] == pytest.approx(1.0)
        assert ORDER_REPLACED in row["settlement_type"]
        assert all(alpha == beta == 1.0 and plays == 0
                   for _arm, alpha, beta, plays in _arms(learning)), \
            "the arm learned from an order it did not choose"
        assert _fact_score(memory, "aaaa") is not None, \
            "the memory the person saw and the outcome named lost its credit"

    def test_an_unmarked_play_still_moves_its_arm(self, store) -> None:
        learning, memory = store
        play_id = _play(learning, ["aaaa"])
        at = datetime.fromisoformat(_row(learning, play_id)["played_at"]) + timedelta(seconds=90)
        _report(memory, ["aaaa"], 1.0, at)
        settle_from_outcomes(_PROFILE, learning, memory, now=at + timedelta(seconds=30))
        (_arm, alpha, beta, plays), = _arms(learning)
        assert alpha == pytest.approx(2.0) and beta == pytest.approx(1.0) and plays == 1

    def test_marking_a_settled_play_changes_nothing(self, store) -> None:
        learning, _memory = store
        play_id = _play(learning, ["aaaa"])
        bandit = ContextualBandit(learning, _PROFILE)
        assert bandit.update(play_id, 1.0)
        assert bandit.mark_order_replaced(play_id) is False
        assert _row(learning, play_id)["settlement_type"] == "proxy_position"


# -- the recall path decides when the order was replaced ---------------------------

@dataclass
class _Fact:
    fact_id: str


@dataclass
class _Result:
    fact: _Fact


@dataclass
class _Response:
    results: list
    reranker_status: str = "applied"


def _results(*ids: str) -> list:
    return [_Result(_Fact(i)) for i in ids]


class TestTheRecallPathMarksOnlyAReplacedOrder:
    def test_a_reorder_that_moved_the_shown_top_marks_the_play(self, store) -> None:
        learning, _memory = store
        play_id = _play(learning, ["a", "b", "c"])
        sink = {"play_id": play_id, "learning_db": str(learning), "shown": ["a", "b", "c"]}
        response = _Response(_results("c", "a", "b"), STATUS_LISTWISE)
        recall_pipeline._note_order_replaced(sink, _PROFILE, response, ["a", "b", "c"])
        assert _row(learning, play_id)["settlement_type"] == ORDER_REPLACED

    @pytest.mark.parametrize("status,ids", [
        ("applied", ("c", "a", "b")),          # a local reorder: the arm's own pipeline
        (STATUS_LISTWISE, ("a", "b", "c")),    # the hosted model kept the arm's order
    ])
    def test_otherwise_the_play_is_left_alone(self, store, status, ids) -> None:
        learning, _memory = store
        play_id = _play(learning, ["a", "b", "c"])
        sink = {"play_id": play_id, "learning_db": str(learning)}
        recall_pipeline._note_order_replaced(sink, _PROFILE, _Response(_results(*ids), status),
                                             ["a", "b", "c"])
        assert _row(learning, play_id)["settlement_type"] is None

    def test_no_play_is_a_no_op(self) -> None:
        recall_pipeline._note_order_replaced({}, _PROFILE,
                                             _Response(_results("b", "a"), STATUS_LISTWISE),
                                             ["a", "b"])


# -- L-11: compare against what the play recorded ----------------------------------

def _shown(learning: Path, play_id: int) -> list[str]:
    raw = _row(learning, play_id)["shown_fact_ids"]
    return json.loads(raw) if raw else []


class TestTheShownSetIsComparedWithWhatWasRecorded:
    def test_a_later_pass_that_moved_a_memory_in_is_caught(self, store) -> None:
        """The exact-lexical guard lifts "z" in after the play was recorded. The
        baseline the old code compared against was taken after that guard, so
        it equalled the final list and the stale record was never corrected."""
        learning, _memory = store
        play_id = _play(learning, ["a", "b", "c"])
        sink = {"play_id": play_id, "learning_db": str(learning), "shown": ["a", "b", "c"]}
        after_guard = _results("z", "a", "b")
        recall_pipeline._resettle_shown_after_bias(sink, _PROFILE, after_guard, ["z", "a", "b"])
        assert set(_shown(learning, play_id)) == {"z", "a", "b"}
        assert sink["shown"] == ["z", "a", "b"], "the sink must keep naming what is recorded"

    def test_the_ranking_pass_leaves_the_recorded_list_in_the_sink(self, store) -> None:
        """``apply_v2_bandit_ensemble`` is where the play is recorded; the sink
        must carry that exact list for the later comparisons."""
        learning, _memory = store
        from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

        response = RecallResponse(results=[
            RetrievalResult(fact=AtomicFact(fact_id=f"f{i}", content=f"m{i}"), score=0.9 - i / 10,
                            confidence=1.0)
            for i in range(6)
        ], query_type="open_domain")
        sink: dict = {}
        recall_pipeline.apply_v2_bandit_ensemble(
            response, "q", _PROFILE, "qid-1", learning_db_path=learning, play_sink=sink)
        assert sink.get("play_id")
        assert sink.get("shown") == _shown(learning, sink["play_id"])


# -- end to end: a hosted reorder, a real play, a real settlement ------------------

def test_a_hosted_reorder_does_not_train_the_arm_end_to_end(
    store, mode_a_config, monkeypatch, tmp_path,
) -> None:
    from types import SimpleNamespace

    from superlocalmemory.core import judge_selection
    from superlocalmemory.core.judge_keys import JudgeKeyStore
    from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult
    from tests.test_core.test_the_answer_check_on_the_recall_path import (
        FAKE_KEY,
        _jev,
        _Provider,
    )

    monkeypatch.setattr(judge_selection, "_live", None)
    learning, memory = store
    keys = JudgeKeyStore(slm_home=tmp_path / "keys")
    keys.set_key("typesafe", FAKE_KEY)
    response = RecallResponse(results=[
        RetrievalResult(fact=AtomicFact(fact_id=f"fact-{i}", content=f"memory {i}"),
                        score=0.9 - i / 100, confidence=1.0)
        for i in range(4)
    ])

    def ranking(resp, query, profile_id, query_id, *, play_sink=None, **_kw):
        bandit = ContextualBandit(learning, _PROFILE)
        choice = bandit.choose({"query_type": "open_domain", "entity_count": 0}, query_id)
        ids = [r.fact.fact_id for r in resp.results[:5]]
        bandit.record_shown(choice.play_id, ids)
        play_sink.update(play_id=choice.play_id, learning_db=str(learning), shown=ids)
        return resp

    monkeypatch.setattr(recall_pipeline, "apply_ranking", ranking)
    engine = SimpleNamespace(_sufficiency_judge=_jev(keys, _Provider(), rerank_k=20),
                             recall=lambda *a, **k: response)
    out = recall_pipeline.run_recall(
        "which one?", _PROFILE, fast=True, config=mode_a_config, retrieval_engine=engine,
        trust_scorer=None, embedder=None, db=SimpleNamespace(db_path=None), llm=None,
        hooks=None)
    assert out.reranker_status == STATUS_LISTWISE
    assert out.results[0].fact.fact_id == "fact-3", "the provider put the last memory first"

    conn = sqlite3.connect(str(learning))
    play_id, played_at = conn.execute("SELECT play_id, played_at FROM bandit_plays").fetchone()
    conn.close()
    at = datetime.fromisoformat(played_at) + timedelta(seconds=90)
    _report(memory, ["fact-3"], 1.0, at)
    assert settle_from_outcomes(_PROFILE, learning, memory, now=at + timedelta(seconds=30)) == 1
    assert all(plays == 0 for *_rest, plays in _arms(learning)), \
        "the arm was rewarded for the hosted model's order"
