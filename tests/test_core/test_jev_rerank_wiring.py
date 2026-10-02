# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Where the optional reordering meets the recall path, and when it stays away.

The reorder is opt-in, inside the online option only, and fails open: every
failure must leave a recall exactly as it would have been without it — the
same memories, in the same order, as the same objects.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import httpx
import pytest

from superlocalmemory.core import engine_wiring, judge_selection, recall_pipeline
from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.core.recall_pipeline import _judge_sufficiency
from superlocalmemory.core.score_contract import finalize_score_contract
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge
from superlocalmemory.retrieval.jev_rerank import CHOICE_KEY, RerankVerdict
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

# The selection fixture fakes platform, runtime, key store and both judges.
from tests.test_core.test_judge_selection import world  # noqa: F401 — a fixture

FAKE_KEY = "sk-test-" + "z9Y8x7W6" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]
_VERDICT = SufficiencyVerdict((0.9,), 0.5, "jev:test")


@pytest.fixture(autouse=True)
def _no_live_judge(monkeypatch):
    monkeypatch.setattr(judge_selection, "_live", None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _response(n: int, *, status: str = "applied") -> RecallResponse:
    return RecallResponse(
        results=[
            RetrievalResult(fact=AtomicFact(fact_id=f"fact-{i:02d}", content=f"memory {i}",
                                            confidence=0.8),
                            score=0.9 - i * 0.01, confidence=1.0)
            for i in range(n)
        ],
        reranker_applied=status == "applied",
        reranker_status=status,
    )


def _ids(response: RecallResponse) -> list[str]:
    return [r.fact.fact_id for r in response.results]


def _ok(answers) -> httpx.Response:
    return httpx.Response(200, json={"model": MODEL, "answers": answers,
                                     "usage": {"input_tokens": 1, "output_tokens": 1}})


def _favouring(*winners: int, noul: float = 0.9):
    """A provider whose choice ranks ``winners`` first, in that order."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        keys = list(body["state"]["memories"])
        if CHOICE_KEY not in body["questions"]:
            return _ok({k: {"type": "noul", "noul": noul} for k in body["questions"]})
        weight = {f"m{w}": float(len(winners) - rank) for rank, w in enumerate(winners)}
        raw = {k: weight.get(k, 0.0) + 0.01 for k in keys}
        total = sum(raw.values())
        probs = {k: v / total for k, v in raw.items()}
        answers = {k: {"type": "noul", "noul": noul} for k in keys}
        answers[CHOICE_KEY] = {"type": "choice", "choice": max(probs, key=probs.get),
                               "probabilities": probs, "confidence": 0.7}
        return _ok(answers)

    return handler


class _Recorder:
    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._wrap(handler))

    def _wrap(self, handler):
        def wrapped(request):
            self.requests.append(request)
            return handler(request)
        return wrapped


@pytest.fixture
def key_store(tmp_path) -> JudgeKeyStore:
    store = JudgeKeyStore(slm_home=tmp_path)
    store.set_key("typesafe", FAKE_KEY)
    return store


def _jev(key_store, recorder, *, rerank_k=10) -> JevSufficiencyJudge:
    return JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                               transport=recorder.transport, rerank_k=rerank_k)


def _engine(judge):
    return SimpleNamespace(_sufficiency_judge=judge)


# ---------------------------------------------------------------------------
# the new order reaches the response
# ---------------------------------------------------------------------------

class TestTheNewOrderIsApplied:
    def test_reordered_within_k_untouched_beyond_and_nothing_dropped(self, key_store):
        recorder = _Recorder(_favouring(7, 3))
        response = _response(25)
        before = list(response.results)

        verdict = _judge_sufficiency(_engine(_jev(key_store, recorder)), "q?", response)

        assert len(recorder.requests) == 1
        ids = _ids(response)
        assert ids[:2] == ["fact-07", "fact-03"]
        assert ids[2:10] == ["fact-00", "fact-01", "fact-02", "fact-04", "fact-05",
                             "fact-06", "fact-08", "fact-09"]
        assert response.results[10:] == before[10:], "beyond k keeps its order"
        assert sorted(ids) == sorted(r.fact.fact_id for r in before), "nothing dropped"
        assert len(ids) == 25
        assert isinstance(verdict, SufficiencyVerdict)

    def test_rank_position_follows_the_new_order(self, key_store):
        response = _response(6)
        verdict = _judge_sufficiency(
            _engine(_jev(key_store, _Recorder(_favouring(5)))), "q?", response)
        finalize_score_contract(response, verdict=verdict)

        assert response.results[0].fact.fact_id == "fact-05"
        assert [r.rank_position for r in response.results] == [1, 2, 3, 4, 5, 6]
        assert response.calibration_status == "measured_public_benchmark_not_calibrated"

    @pytest.mark.parametrize("local", ["applied", "fallback_not_ready", "not_configured"])
    def test_a_reordered_response_says_so_and_keeps_the_local_status(self, key_store, local):
        response = _response(5, status=local)
        _judge_sufficiency(_engine(_jev(key_store, _Recorder(_favouring(2)))), "q?", response)

        assert response.reranker_applied is True
        assert response.reranker_status == "jev_listwise"
        assert response.local_reranker_status == local

    def test_the_verdict_is_on_the_top_three_the_caller_sees(self, key_store):
        def handler(request):
            body = json.loads(request.content)
            keys = list(body["state"]["memories"])
            probs = {k: 0.01 for k in keys}
            probs["m4"] = 1 - 0.01 * (len(keys) - 1)
            answers = {k: {"type": "noul", "noul": 0.05} for k in keys}
            answers["m4"] = {"type": "noul", "noul": 0.97}
            answers[CHOICE_KEY] = {"type": "choice", "choice": "m4",
                                   "probabilities": probs, "confidence": 0.9}
            return _ok(answers)

        response = _response(6)
        verdict = _judge_sufficiency(_engine(_jev(key_store, _Recorder(handler))), "q?", response)
        finalize_score_contract(response, verdict=verdict)

        assert verdict.probabilities == (0.97, 0.05, 0.05)
        assert response.abstained is False
        assert response.answer_confidence == 0.97


# ---------------------------------------------------------------------------
# failing open
# ---------------------------------------------------------------------------

def _timeout(request):
    raise httpx.ReadTimeout("slow", request=request)


def _malformed(request):
    keys = list(json.loads(request.content)["state"]["memories"])
    return _ok({k: {"type": "noul", "noul": 0.9} for k in keys})  # no choice answer


class TestEveryFailureLeavesTheRecallAsItWas:
    @pytest.mark.parametrize("handler", [
        _timeout,
        lambda r: httpx.Response(503, json={"error": "x"}),
        lambda r: httpx.Response(200, content=b"oops"),
        _malformed,
    ], ids=["timeout", "http_503", "not_json", "malformed"])
    def test_same_memories_same_order_same_objects(self, key_store, handler):
        recorder = _Recorder(handler)
        response = _response(12, status="applied")
        before = list(response.results)

        verdict = _judge_sufficiency(_engine(_jev(key_store, recorder)), "q?", response)

        assert verdict is None
        assert len(recorder.requests) == 1, "no second request after a failed one"
        assert all(a is b for a, b in zip(response.results, before, strict=True))
        assert response.reranker_applied is True
        assert response.reranker_status == "applied"
        assert response.local_reranker_status == ""

    def test_a_judge_that_raises_never_breaks_the_recall(self):
        class _Raising:
            backend = "jev"
            top_k = 3
            rerank_enabled = True
            rerank_k = 10

            def rerank_and_judge(self, query, documents):
                raise RuntimeError("bug")

        response = _response(5)
        before = _ids(response)
        assert _judge_sufficiency(_engine(_Raising()), "q?", response) is None
        assert _ids(response) == before

    @pytest.mark.parametrize("order", [
        (0, 0, 1, 2, 3),          # a duplicate would drop a memory
        (0, 1, 2, 3, 4, 5),       # longer than the results
        (1, 2, 3, 4, 5),          # not a permutation of its own positions
        (-1, 0, 1, 2, 3),
        ("0", "1"),
        (),
    ])
    def test_an_order_that_is_not_a_permutation_is_never_applied(self, order):
        class _Bad:
            backend = "jev"
            top_k = 3
            rerank_enabled = True
            rerank_k = 10

            def rerank_and_judge(self, query, documents):
                return RerankVerdict(order=order, verdict=_VERDICT)

        response = _response(5)
        before = _ids(response)
        assert _judge_sufficiency(_engine(_Bad()), "q?", response) is None
        assert _ids(response) == before
        assert response.reranker_status == "applied"

    @pytest.mark.parametrize("outcome", [_VERDICT, None, "reordered", (1, 0)])
    def test_anything_but_a_rerank_verdict_is_ignored(self, outcome):
        class _Odd:
            backend = "jev"
            top_k = 3
            rerank_enabled = True
            rerank_k = 10

            def rerank_and_judge(self, query, documents):
                return outcome

        response = _response(5)
        before = _ids(response)
        assert _judge_sufficiency(_engine(_Odd()), "q?", response) is None
        assert _ids(response) == before

    @pytest.mark.parametrize("rerank_k", [0, -1, True, "10", None])
    def test_a_judge_with_a_nonsense_count_is_not_asked(self, rerank_k):
        calls = []

        class _Nonsense:
            backend = "jev"
            top_k = 3
            rerank_enabled = True

            def rerank_and_judge(self, query, documents):
                calls.append(len(documents))
                return RerankVerdict(order=(1, 0), verdict=_VERDICT)

        judge = _Nonsense()
        judge.rerank_k = rerank_k
        response = _response(5)
        before = _ids(response)
        assert _judge_sufficiency(_engine(judge), "q?", response) is None
        assert calls == [] and _ids(response) == before

    def test_background_work_sends_nothing(self, key_store):
        from superlocalmemory.core.recall_gate import background_work

        recorder = _Recorder(_favouring(3))
        response = _response(6)
        before = _ids(response)
        with background_work():
            assert _judge_sufficiency(_engine(_jev(key_store, recorder)), "q?", response) is None
        assert recorder.requests == []
        assert _ids(response) == before


class TestOnlyTheOnlineOptionEverReorders:
    def test_reordering_off_sends_the_plain_check(self, key_store):
        recorder = _Recorder(_favouring(3))
        response = _response(6)
        before = _ids(response)

        verdict = _judge_sufficiency(_engine(_jev(key_store, recorder, rerank_k=0)), "q?",
                                     response)

        body = json.loads(recorder.requests[0].content)
        assert CHOICE_KEY not in body["questions"]
        assert list(body["questions"]) == ["m0", "m1", "m2"]
        assert _ids(response) == before
        assert isinstance(verdict, SufficiencyVerdict)
        assert response.reranker_status == "applied"

    def test_a_local_judge_never_reorders_even_if_it_claims_to(self):
        calls = []

        class _LocalClaimingToReorder:
            backend = "laya"
            top_k = 3
            rerank_enabled = True
            rerank_k = 10

            def judge(self, query, documents):
                calls.append(("judge", len(documents)))
                return _VERDICT

            def rerank_and_judge(self, query, documents):
                calls.append(("rerank", len(documents)))
                return RerankVerdict(order=(1, 0), verdict=_VERDICT)

        response = _response(5)
        before = _ids(response)
        assert _judge_sufficiency(_engine(_LocalClaimingToReorder()), "q?", response) is _VERDICT
        assert calls == [("judge", 3)]
        assert _ids(response) == before

    def test_rerank_enabled_must_be_the_boolean_true(self):
        calls = []

        class _Truthy:
            backend = "jev"
            top_k = 3
            rerank_enabled = "yes"
            rerank_k = 10

            def judge(self, query, documents):
                calls.append("judge")
                return _VERDICT

            def rerank_and_judge(self, query, documents):
                calls.append("rerank")

        _judge_sufficiency(_engine(_Truthy()), "q?", _response(5))
        assert calls == ["judge"]


# ---------------------------------------------------------------------------
# selection: who may reorder
# ---------------------------------------------------------------------------

def _cfg(mode: str, **kw) -> RetrievalConfig:
    return RetrievalConfig(sufficiency_judge=mode, **kw)


_VALUES = [True, False, "true", 1, None]


class TestTheGate:
    @pytest.mark.parametrize("on,consent", list(itertools.product(_VALUES, _VALUES)))
    def test_both_flags_must_be_the_boolean_true(self, on, consent):
        cfg = _cfg("jev", sufficiency_jev_rerank=on, sufficiency_jev_rerank_consent=consent)
        k = judge_selection.jev_rerank_k(cfg)
        assert (k > 0) is (on is True and consent is True)

    @pytest.mark.parametrize("asked,expected", [
        # F10: a malformed count means reordering OFF, never "send twenty".
        (3, 5), (5, 5), (12, 12), (30, 30), (50, 30), ("abc", 0), (None, 0), (True, 0),
    ])
    def test_the_count_is_clamped_and_a_bad_value_means_off(self, asked, expected):
        cfg = _cfg("jev", sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True,
                   sufficiency_jev_rerank_k=asked)
        assert judge_selection.jev_rerank_k(cfg) == expected

    def test_off_by_default(self):
        assert RetrievalConfig().sufficiency_jev_rerank is False
        assert RetrievalConfig().sufficiency_jev_rerank_consent is False
        assert RetrievalConfig().sufficiency_jev_rerank_k == 20
        assert judge_selection.jev_rerank_k(RetrievalConfig()) == 0


class TestWhoIsBuiltWithReordering:
    def _both(self, mode, **kw):
        kw.setdefault("sufficiency_jev_consent", True)
        return _cfg(mode, sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True,
                    **kw)

    def test_the_online_check_gets_the_count(self, world):  # noqa: F811
        judge = engine_wiring.init_sufficiency_judge(self._both("jev"))
        assert judge.kwargs["rerank_k"] == 20

    def test_without_its_own_consent_the_online_check_does_not_reorder(self, world):  # noqa: F811
        cfg = _cfg("jev", sufficiency_jev_consent=True, sufficiency_jev_rerank=True,
                   sufficiency_jev_rerank_consent=False)
        assert engine_wiring.init_sufficiency_judge(cfg).kwargs["rerank_k"] == 0

    @pytest.mark.parametrize("mode", ["laya", "auto"])
    def test_the_local_check_is_never_built_to_reorder(self, world, mode):  # noqa: F811
        judge = engine_wiring.init_sufficiency_judge(self._both(mode))
        assert judge_selection.backend_of(judge) == "laya"
        assert "rerank_k" not in judge.kwargs
        assert world.events == ["laya.init"]

    def test_auto_never_starts_the_online_check_to_reorder(self, world):  # noqa: F811
        world.supported = False
        assert engine_wiring.init_sufficiency_judge(self._both("auto")) is None
        assert world.events == [] and world.stores == []

    def test_off_is_off(self, world):  # noqa: F811
        assert engine_wiring.init_sufficiency_judge(self._both("off")) is None

    def test_without_a_key_nothing_is_built(self, world):  # noqa: F811
        world.keys = set()
        assert engine_wiring.init_sufficiency_judge(self._both("jev")) is None

    def test_without_the_answer_check_consent_nothing_is_built(self, world):  # noqa: F811
        cfg = self._both("jev", sufficiency_jev_consent=False)
        assert engine_wiring.init_sufficiency_judge(cfg) is None


# ---------------------------------------------------------------------------
# the whole recall: everything downstream sees the final order
# ---------------------------------------------------------------------------

class TestTheWholeRecallSeesTheFinalOrder:
    def test_markers_positions_working_memory_and_play_evidence(
        self, key_store, mode_a_config, monkeypatch,
    ):
        response = _response(8)
        judge = _jev(key_store, _Recorder(_favouring(6, 5)))
        engine = SimpleNamespace(_sufficiency_judge=judge,
                                 recall=lambda *a, **kw: response)
        admitted, resettled = [], []

        def fake_ranking(resp, *a, play_sink=None, **kw):
            play_sink["play_id"] = "play-1"
            play_sink["learning_db"] = "/nowhere/learning.db"
            return resp

        monkeypatch.setattr(recall_pipeline, "apply_ranking", fake_ranking)
        monkeypatch.setattr(recall_pipeline, "_admit_to_working_memory",
                            lambda results, pid, sid: admitted.append(
                                [r.fact.fact_id for r in results]))
        monkeypatch.setattr(recall_pipeline, "_resettle_shown_after_bias",
                            lambda sink, pid, results, before: resettled.append(
                                (sink.get("play_id"), [r.fact.fact_id for r in results[:5]],
                                 list(before))))

        out = recall_pipeline.run_recall(
            "which one?", "default", session_id="conversation-42", fast=True,
            config=mode_a_config, retrieval_engine=engine, trust_scorer=None,
            embedder=None, db=SimpleNamespace(db_path=None), llm=None, hooks=None,
        )

        ids = [r.fact.fact_id for r in out.results]
        assert ids[:2] == ["fact-06", "fact-05"]
        assert [r.rank_position for r in out.results] == list(range(1, 9))
        assert all(r.marker.startswith(f"slm:fact:{r.fact.fact_id}:") for r in out.results)
        assert admitted == [ids]
        assert resettled[-1] == ("play-1", ids[:5],
                                 ["fact-00", "fact-01", "fact-02", "fact-03", "fact-04"])
        assert out.reranker_status == "jev_listwise"


# ---------------------------------------------------------------------------
# a caller on the wire can tell who ordered the answer
# ---------------------------------------------------------------------------

class TestTheSerialisedAnswerSaysWhoOrderedIt:
    """MCP, CLI and HTTP all serialise through ``recall_response_metadata``.
    The online model is not perfectly repeatable, so a caller comparing two
    runs must be able to see that the order came from it."""

    def test_a_reordered_answer_says_so_and_keeps_the_local_status(self):
        from superlocalmemory.server.recall_serializer import recall_response_metadata

        response = RecallResponse(query="q", reranker_applied=True,
                                  reranker_status="jev_listwise",
                                  local_reranker_status="applied")
        meta = recall_response_metadata(response)
        assert meta["reranker_status"] == "jev_listwise"
        assert meta["local_reranker_status"] == "applied"

    def test_an_answer_ordered_locally_reports_the_local_status_only(self):
        from superlocalmemory.server.recall_serializer import recall_response_metadata

        meta = recall_response_metadata(RecallResponse(query="q", reranker_status="applied"))
        assert meta["reranker_status"] == "applied"
        assert meta["local_reranker_status"] == ""

    def test_an_older_response_without_the_fields_still_serialises(self):
        from superlocalmemory.server.recall_serializer import recall_response_metadata

        class _Legacy:
            query = "q"

        meta = recall_response_metadata(_Legacy())
        assert meta["reranker_status"] == "not_configured"
        assert meta["local_reranker_status"] == ""
