# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The answer check on the recall path: who may ask it, how long it may take,
and what every response says about it.

* A recall the system makes on its own (the daemon warm-up) never asks the
  check — no hosted request, no bill, no memories leaving the machine.
* The check gets what is left of the recall's 2.0 s ceiling, and below a floor
  it is not asked at all. Skipping it costs a verdict, never a result.
* Every response says what became of the check, so a caller comparing two runs
  can tell "not judged this time" from "nothing answers".
* The bounded-loop gate asks the check but never the reorder.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from superlocalmemory.core import judge_selection, recall_pipeline
from superlocalmemory.core.answer_check_scope import skip_answer_check
from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge
from superlocalmemory.retrieval.jev_rerank import CHOICE_KEY
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.server.recall_serializer import recall_response_metadata
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

FAKE_KEY = "sk-test-" + "q1W2e3R4" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]
_VERDICT = SufficiencyVerdict((0.9,), 0.5, "test:judge")


@pytest.fixture(autouse=True)
def _no_live_judge(monkeypatch):
    monkeypatch.setattr(judge_selection, "_live", None)


class _Judge:
    """A local-style judge that records every question it is asked."""

    backend = "laya"
    top_k = 3

    def __init__(self, verdict=_VERDICT, status=acs.STATUS_JUDGED) -> None:
        self.verdict, self.status = verdict, status
        self.calls: list[tuple] = []

    def assess(self, query, documents, *, deadline=None):
        self.calls.append((query, len(documents), deadline))
        return acs.JudgeOutcome(self.verdict, self.status)

    def judge(self, query, documents):
        return self.assess(query, documents).verdict

    def shutdown(self) -> None: ...


def _response(n: int) -> RecallResponse:
    return RecallResponse(results=[
        RetrievalResult(fact=AtomicFact(fact_id=f"fact-{i:02d}", content=f"memory {i}",
                                        confidence=0.8),
                        score=0.9 - i * 0.01, confidence=1.0)
        for i in range(n)
    ], reranker_applied=True, reranker_status="applied")


def _run(judge, config, monkeypatch, *, n: int = 5, delay: float = 0.0, **kw):
    response = _response(n)

    def recall(*_a, **_kw):
        if delay:
            time.sleep(delay)
        return response

    engine = SimpleNamespace(_sufficiency_judge=judge, recall=recall)
    monkeypatch.setattr(recall_pipeline, "apply_ranking", lambda resp, *a, **k: resp)
    return recall_pipeline.run_recall(
        "which one?", "default", fast=True, config=config, retrieval_engine=engine,
        trust_scorer=None, embedder=None, db=SimpleNamespace(db_path=None), llm=None,
        hooks=None, **kw)


def _ids(response) -> list[str]:
    return [r.fact.fact_id for r in response.results]


# -- a hosted judge that counts what it is sent ----------------------------------

class _Provider:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        keys = list(body["state"]["memories"])
        answers = {k: {"type": "noul", "noul": 0.9 if k == "m0" else 0.2} for k in keys}
        if CHOICE_KEY in body["questions"]:
            share = 1.0 / len(keys)
            probs = {k: share for k in keys}
            probs[keys[-1]] += 0.5
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            answers[CHOICE_KEY] = {"type": "choice", "choice": keys[-1],
                                   "probabilities": probs}
        return httpx.Response(200, json={"model": MODEL, "answers": answers, "usage": {}})


@pytest.fixture
def key_store(tmp_path) -> JudgeKeyStore:
    store = JudgeKeyStore(slm_home=tmp_path)
    store.set_key("typesafe", FAKE_KEY)
    return store


def _jev(key_store, provider, *, rerank_k: int = 0) -> JevSufficiencyJudge:
    return JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                               transport=provider.transport, rerank_k=rerank_k)


# ---------------------------------------------------------------------------
# S-H1: the system's own recalls are never judged
# ---------------------------------------------------------------------------

class TestASystemRecallNeverAsksTheCheck:
    def test_no_question_is_put_to_the_judge(self, mode_a_config, monkeypatch) -> None:
        judge = _Judge()
        with skip_answer_check():
            out = _run(judge, mode_a_config, monkeypatch)
        assert judge.calls == []
        assert out.answer_check_status == acs.STATUS_SKIPPED
        assert out.answer_confidence is None and out.abstained is False
        assert _ids(out) == [f"fact-{i:02d}" for i in range(5)], "results are untouched"

    def test_no_hosted_request_is_sent_and_nothing_is_billed(
        self, mode_a_config, monkeypatch, key_store,
    ) -> None:
        provider = _Provider()
        with skip_answer_check():
            out = _run(_jev(key_store, provider, rerank_k=20), mode_a_config, monkeypatch)
        assert provider.bodies == [], "a warm-up recall sent memories to the provider"
        assert out.reranker_status == "applied", "nothing reordered it either"

    def test_a_caller_recall_still_asks(self, mode_a_config, monkeypatch, key_store) -> None:
        provider = _Provider()
        out = _run(_jev(key_store, provider), mode_a_config, monkeypatch)
        assert len(provider.bodies) == 1
        assert out.answer_check_status == acs.STATUS_JUDGED

    def test_an_unknown_request_is_refused_not_run_in_full(
        self, mode_a_config, monkeypatch,
    ) -> None:
        for value in ("sometimes", "skip"):  # skipping is the shared marker, not a value
            with pytest.raises(ValueError):
                _run(_Judge(), mode_a_config, monkeypatch, answer_check=value)


# ---------------------------------------------------------------------------
# M-3: the check gets what is left of the recall's ceiling
# ---------------------------------------------------------------------------

class TestTheCheckFitsTheRecallBudget:
    def test_it_is_given_the_time_left_under_the_ceiling(
        self, mode_a_config, monkeypatch,
    ) -> None:
        judge = _Judge()
        before = time.monotonic()
        _run(judge, mode_a_config, monkeypatch)
        after = time.monotonic()
        (_q, _n, deadline), = judge.calls
        assert deadline is not None, "the judge was not told how long it may take"
        ceiling = acs.RECALL_CEILING_S - acs.POST_JUDGE_RESERVE_S
        assert before + ceiling - 0.01 <= deadline <= after + ceiling

    def test_too_little_time_left_skips_the_check_and_keeps_every_result(
        self, mode_a_config, monkeypatch,
    ) -> None:
        # Shrink the ceiling instead of sleeping two seconds: retrieval takes
        # 0.2 s of a 0.4 s ceiling, leaving less than the floor.
        monkeypatch.setattr(acs, "RECALL_CEILING_S", 0.4)
        judge = _Judge()
        out = _run(judge, mode_a_config, monkeypatch, delay=0.2)
        assert judge.calls == []
        assert out.answer_check_status == acs.STATUS_SKIPPED
        assert _ids(out) == [f"fact-{i:02d}" for i in range(5)]
        assert out.abstained is False and out.answer_confidence is None


# ---------------------------------------------------------------------------
# M-5: every response says what became of the check
# ---------------------------------------------------------------------------

class TestEveryResponseNamesWhatBecameOfTheCheck:
    @pytest.mark.parametrize("status", [acs.STATUS_BUSY, acs.STATUS_WARMING,
                                        acs.STATUS_UNAVAILABLE])
    def test_an_unanswered_check_says_why(self, status, mode_a_config, monkeypatch) -> None:
        out = _run(_Judge(verdict=None, status=status), mode_a_config, monkeypatch)
        assert out.answer_check_status == status
        assert recall_response_metadata(out)["answer_check_status"] == status
        assert out.answer_confidence is None

    def test_a_judged_recall_says_judged(self, mode_a_config, monkeypatch) -> None:
        out = _run(_Judge(), mode_a_config, monkeypatch)
        assert recall_response_metadata(out)["answer_check_status"] == acs.STATUS_JUDGED
        assert out.answer_confidence == 0.9

    def test_no_judge_says_off(self, mode_a_config, monkeypatch) -> None:
        out = _run(None, mode_a_config, monkeypatch)
        assert recall_response_metadata(out)["answer_check_status"] == acs.STATUS_OFF

    def test_nothing_to_judge_says_skipped(self, mode_a_config, monkeypatch) -> None:
        judge = _Judge()
        out = _run(judge, mode_a_config, monkeypatch, n=0)
        assert judge.calls == []
        assert out.answer_check_status == acs.STATUS_SKIPPED

    def test_a_judge_without_a_status_still_reports_one(
        self, mode_a_config, monkeypatch,
    ) -> None:
        """Third-party or older judges implement only ``judge``."""

        class _Plain:
            backend, top_k = "laya", 3

            def judge(self, query, documents):
                return None

            def shutdown(self) -> None: ...

        out = _run(_Plain(), mode_a_config, monkeypatch)
        assert out.answer_check_status == acs.STATUS_UNAVAILABLE

    def test_the_status_survives_serialisation_for_a_response_built_elsewhere(self) -> None:
        assert recall_response_metadata(RecallResponse())["answer_check_status"] in \
            acs.ANSWER_CHECK_STATUSES


# ---------------------------------------------------------------------------
# MU-2: the loop gate asks the check, never the reorder
# ---------------------------------------------------------------------------

class TestTheGateAsksTheCheckButNeverTheReorder:
    def test_no_reorder_sends_the_plain_check_only(
        self, mode_a_config, monkeypatch, key_store,
    ) -> None:
        provider = _Provider()
        out = _run(_jev(key_store, provider, rerank_k=20), mode_a_config, monkeypatch,
                   n=3, answer_check=acs.REQUEST_NO_REORDER)
        assert len(provider.bodies) == 1, "one request per lap, never more"
        (body,) = provider.bodies
        assert CHOICE_KEY not in body["questions"], "the gate paid for a reorder"
        assert len(body["state"]["memories"]) <= 3
        assert _ids(out) == ["fact-00", "fact-01", "fact-02"], "the order is untouched"
        assert out.reranker_status == "applied"
        assert out.answer_check_status == acs.STATUS_JUDGED

    def test_a_full_recall_still_reorders(self, mode_a_config, monkeypatch, key_store) -> None:
        provider = _Provider()
        out = _run(_jev(key_store, provider, rerank_k=20), mode_a_config, monkeypatch, n=3)
        assert CHOICE_KEY in provider.bodies[0]["questions"]
        assert out.reranker_status == "jev_listwise"


def test_the_recall_ceiling_is_the_owners_three_seconds() -> None:
    # The owner's rule (2026-10-03): retrieval plus answer check within 3.0 s.
    # Changing it is the owner's decision, not a tuning knob.
    assert acs.RECALL_CEILING_S == 3.0
