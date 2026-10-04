# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The same question gets the same answer-check outcome, and a skip says why (WP14).

Seen on a loaded machine: one question asked three times came back "skipped",
"skipped", "judged" — retrieval sometimes used the whole 3.0 s budget, and the
skip looked like a recall that answered (``abstained`` False, no reason). Each
test pins one part of the fix, against fake judges (no model is loaded).
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from superlocalmemory.core import answer_check_memo as memo
from superlocalmemory.core import judge_selection, recall_pipeline
from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval import sufficiency as laya_mod
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge
from superlocalmemory.retrieval.sufficiency import LayaSufficiencyJudge, SufficiencyVerdict
from superlocalmemory.server.recall_serializer import recall_response_metadata
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

FAKE_KEY = "sk-test-" + "z9Y8x7W6" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]


@pytest.fixture(autouse=True)
def _no_live_judge(monkeypatch):
    monkeypatch.setattr(judge_selection, "_live", None)


#: Simulated retrieval times against the production 3.0 s ceiling. A check is
#: skipped once retrieval has used more than 2.7 s, so SLOW is well past that
#: and FAST leaves 2.7 s of headroom for real overhead on a loaded machine.
SLOW_S = 2.8
FAST_S = 0.0


class _RetrievalClock:
    """The clock the answer-check budget reads, ahead of real time by however
    long the fake retrieval "took". No real sleep, so no load can move a run
    across the skip line. (Real sleeps against a 0.6 s ceiling left 0.25 s of
    headroom; under load a "fast" run crossed it and was skipped.)"""

    def __init__(self) -> None:
        self.offset = 0.0
        self._real = time.monotonic

    def monotonic(self) -> float:
        return self._real() + self.offset


@pytest.fixture(autouse=True)
def retrieval_clock(monkeypatch):
    clock = _RetrievalClock()
    monkeypatch.setattr(acs, "time", SimpleNamespace(monotonic=clock.monotonic,
                                                      clock=clock))
    return clock


# -- fakes -------------------------------------------------------------------------

class _JitteryLaya:
    """An on-device-style judge: deterministic verdict, random latency."""

    backend = "laya"
    top_k = 3
    threshold = 0.5
    calibration_id = "laya:test"

    def __init__(self, seed: int = 7, confidence: float = 0.42) -> None:
        self.rng = random.Random(seed)
        self.confidence = confidence
        self.ready = True
        self.closed = False
        self.asks = 0
        self.idle_asks = 0

    def _verdict(self):
        return SufficiencyVerdict((self.confidence,), self.threshold, self.calibration_id)

    def assess(self, query, documents, *, deadline=None):
        self.asks += 1
        time.sleep(self.rng.uniform(0.0, 0.05))
        return acs.JudgeOutcome(self._verdict(), acs.STATUS_JUDGED)

    def assess_if_idle(self, query, documents):
        self.idle_asks += 1
        return acs.JudgeOutcome(self._verdict(), acs.STATUS_JUDGED)

    def shutdown(self) -> None: ...


def _response(contents) -> RecallResponse:
    return RecallResponse(results=[
        RetrievalResult(fact=AtomicFact(fact_id=f"f{i}", content=c, confidence=0.8),
                        score=0.9 - i * 0.01, confidence=1.0)
        for i, c in enumerate(contents)])


def _run(judge, config, monkeypatch, *, delay: float = FAST_S,
         contents=("memory 0", "memory 1", "memory 2", "memory 3")):
    clock = acs.time.clock  # the autouse retrieval_clock

    def recall(*_a, **_kw):
        clock.offset = delay  # retrieval "took" delay seconds
        return _response(contents)

    engine = SimpleNamespace(_sufficiency_judge=judge, recall=recall)
    monkeypatch.setattr(recall_pipeline, "apply_ranking", lambda resp, *a, **k: resp)
    try:
        return recall_pipeline.run_recall(
            "what is the recall ceiling?", "default", fast=True, config=config,
            retrieval_engine=engine, trust_scorer=None, embedder=None,
            db=SimpleNamespace(db_path=None), llm=None, hooks=None)
    finally:
        clock.offset = 0.0


def _join_finishers(timeout: float = 5.0) -> None:
    for t in threading.enumerate():
        if t.name == "answer-check-finish-later":  # answer_check_deferred.THREAD_NAME
            t.join(timeout)


# -- budget arithmetic at 3.0 s -----------------------------------------------------

class TestTheBudgetIsThreeSeconds:
    def test_the_ceiling_is_the_owners(self) -> None:
        assert acs.RECALL_CEILING_S == 3.0

    def test_the_check_is_asked_until_2_70_s_and_not_after(self) -> None:
        start = 100.0
        assert acs.judge_deadline(start, now=start + 2.69) == pytest.approx(start + 2.95)
        assert acs.judge_deadline(start, now=start + 2.71) is None
        # Vacuity: the old 2.0 s ceiling would already skip at 1.8 s.
        assert acs.judge_deadline(start, now=start + 1.8) is not None


# -- a skip never looks like an answer ---------------------------------------------

class TestASkipSaysSoAndWhy:
    def test_budget_skip_is_on_the_envelope(self, mode_a_config, monkeypatch) -> None:
        judge = _JitteryLaya()
        judge.ready = False  # nothing finished later: this test is about the skip
        out = _run(judge, mode_a_config, monkeypatch, delay=SLOW_S)
        meta = recall_response_metadata(out)
        assert meta["answer_check_status"] == "skipped"
        assert meta["answer_check_ran"] is False
        assert meta["answer_check_reason"] == "budget"
        assert "did not run" in meta["answer_check_note"]
        assert meta["abstained"] is False and meta["answer_confidence"] is None
        assert judge.asks == 0

    def test_cli_prints_the_reason(self) -> None:
        from superlocalmemory.cli.commands import _answer_check_line
        line = _answer_check_line({"calibration_status": "uncalibrated",
                                   "answer_check_status": "skipped",
                                   "answer_check_reason": "budget",
                                   "abstained": False, "answer_confidence": None})
        assert line.startswith("Answer check did not run") and "time budget" in line
        assert "likely answered" not in line

    @pytest.mark.parametrize("status", ["busy", "warming", "unavailable"])
    def test_every_unjudged_status_has_a_note(self, status) -> None:
        assert acs.answer_check_note(status).startswith("Answer check did not run")

    def test_judged_and_off_have_no_note(self) -> None:
        assert acs.answer_check_note("judged") == ""
        assert acs.answer_check_note("judged", acs.DETAIL_REUSED) == ""
        assert acs.answer_check_note("off") == ""


# -- CRIT 1: nobody with no check configured sees anything new ----------------------

class TestNoCheckConfiguredIsUnchanged:
    def test_off_is_silent_everywhere(self, mode_a_config, monkeypatch) -> None:
        from superlocalmemory.cli.commands import _answer_check_line
        out = _run(None, mode_a_config, monkeypatch)
        meta = recall_response_metadata(out)
        assert meta["answer_check_status"] == "off" and meta["answer_check_note"] == ""
        assert _answer_check_line(meta) == ""

    def test_keyword_fallback_without_reason_stays_silent_in_text(self) -> None:
        from superlocalmemory.cli.commands import _answer_check_line
        meta = recall_response_metadata(RecallResponse(results=[]))
        assert _answer_check_line(meta) == ""


# -- the same question, N runs, one outcome -----------------------------------------

def _statuses(judge, config, monkeypatch, delays):
    seen = []
    for d in delays:
        out = _run(judge, config, monkeypatch, delay=d)
        _join_finishers()
        seen.append((out.answer_check_status, out.answer_confidence))
    return seen


class TestTheSameQuestionGetsTheSameVerdict:
    #: Retrieval jitter straddling the budget, as on a loaded machine.
    DELAYS = [2.80, 0.05, 2.82, 0.0, 2.85, 0.02, 2.81, 0.03, 2.84, 0.01]

    def test_after_the_first_run_every_run_is_judged_alike(
            self, mode_a_config, monkeypatch) -> None:
        judge = _JitteryLaya()
        seen = _statuses(judge, mode_a_config, monkeypatch, self.DELAYS)
        assert seen[0] == ("skipped", None)          # honest: no time on run 1
        assert set(seen[1:]) == {("judged", 0.42)}   # then always the same verdict
        # One question, each of its 3 memories asked once (a check finished
        # later asks one memory at a time, so a live recall can cut in).
        assert judge.asks == 0 and judge.idle_asks == 3

    def test_vacuity_without_the_memo_runs_disagree(
            self, mode_a_config, monkeypatch) -> None:
        monkeypatch.setattr(memo, "lookup", lambda *a, **k: None)
        monkeypatch.setattr(memo, "finish_later", lambda *a, **k: None)
        seen = _statuses(_JitteryLaya(), mode_a_config, monkeypatch, self.DELAYS)
        assert {s for s, _ in seen} == {"skipped", "judged"}

    def test_reuse_is_reported(self, mode_a_config, monkeypatch) -> None:
        judge = _JitteryLaya()
        first = _run(judge, mode_a_config, monkeypatch)
        second = _run(judge, mode_a_config, monkeypatch)
        assert first.answer_check_detail == "" and second.answer_check_detail == "reused"
        assert recall_response_metadata(second)["answer_check_ran"] is True
        assert judge.asks == 1

    def test_a_changed_memory_is_judged_afresh(self, mode_a_config, monkeypatch) -> None:
        judge = _JitteryLaya()
        _run(judge, mode_a_config, monkeypatch)
        _run(judge, mode_a_config, monkeypatch,
             contents=("memory 0 (edited)", "memory 1", "memory 2", "memory 3"))
        _run(judge, mode_a_config, monkeypatch,
             contents=("memory 1", "memory 0", "memory 2", "memory 3"))
        # Fourth result is outside top-3: changing it changes nothing judged.
        _run(judge, mode_a_config, monkeypatch,
             contents=("memory 0", "memory 1", "memory 2", "something else"))
        assert judge.asks == 3

    def test_a_new_judge_starts_empty(self, mode_a_config, monkeypatch) -> None:
        a, b = _JitteryLaya(), _JitteryLaya(confidence=0.9)
        _run(a, mode_a_config, monkeypatch)
        out = _run(b, mode_a_config, monkeypatch)
        assert out.answer_confidence == 0.9 and b.asks == 1


# -- CRIT 3: bounded on a large store / long session ---------------------------------

class TestTheMemoIsBounded:
    def test_entries_are_capped_and_expire(self, monkeypatch) -> None:
        monkeypatch.setattr(memo, "MAX_ENTRIES", 4)
        judge = _JitteryLaya()
        v = judge._verdict()
        for i in range(10):
            memo.store(judge, f"q{i}", ["doc"], v, now=0.0)
        assert memo.lookup(judge, "q0", ["doc"], now=1.0) is None
        assert memo.lookup(judge, "q9", ["doc"], now=1.0) is v
        assert memo.lookup(judge, "q9", ["doc"], now=memo.TTL_S + 2.0) is None


# -- one provider at a time ----------------------------------------------------------

class _Provider:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        answers = {k: {"type": "noul", "noul": 0.3} for k in body["state"]["memories"]}
        return httpx.Response(200, json={"model": MODEL, "answers": answers, "usage": {}})


@pytest.fixture
def jev(tmp_path):
    store = JudgeKeyStore(slm_home=tmp_path)
    store.set_key("typesafe", FAKE_KEY)
    provider = _Provider()
    return JevSufficiencyJudge(provider="typesafe", key_store=store,
                               transport=provider.transport), provider


def _forbid(monkeypatch, cls, names, touched):
    for name in names:
        def _spy(self, *a, _n=name, **k):
            touched.append(_n)
            raise AssertionError(f"unselected provider touched: {cls.__name__}.{_n}")
        monkeypatch.setattr(cls, name, _spy, raising=False)


class TestOnlyTheSelectedProviderIsEverTouched:
    def test_jev_selected_never_touches_laya(self, mode_a_config, monkeypatch, jev) -> None:
        judge, provider = jev
        touched: list[str] = []
        _forbid(monkeypatch, LayaSufficiencyJudge,
                ["__init__", "assess", "assess_if_idle", "judge", "start_warmup"], touched)
        ok = _run(judge, mode_a_config, monkeypatch)                # judged
        again = _run(judge, mode_a_config, monkeypatch)             # not memoised
        skipped = _run(judge, mode_a_config, monkeypatch, delay=SLOW_S)  # budget skip
        _join_finishers()
        assert ok.answer_check_status == again.answer_check_status == "judged"
        assert again.answer_check_detail == ""  # the hosted check is asked each time
        assert skipped.answer_check_status == "skipped"
        assert skipped.answer_check_detail == "budget"
        assert len(provider.bodies) == 2        # nothing finished later, nothing billed
        assert touched == []

    def test_jev_failure_is_not_retried_on_laya(self, mode_a_config, monkeypatch,
                                                tmp_path) -> None:
        touched: list[str] = []
        _forbid(monkeypatch, LayaSufficiencyJudge,
                ["__init__", "assess", "assess_if_idle", "judge", "start_warmup"], touched)
        store = JudgeKeyStore(slm_home=tmp_path)
        store.set_key("typesafe", FAKE_KEY)
        down = httpx.MockTransport(lambda r: httpx.Response(503))
        judge = JevSufficiencyJudge(provider="typesafe", key_store=store, transport=down)
        out = _run(judge, mode_a_config, monkeypatch)
        assert out.answer_check_status == "unavailable" and touched == []
        assert "did not run" in recall_response_metadata(out)["answer_check_note"]

    def test_laya_selected_never_touches_jev(self, mode_a_config, monkeypatch) -> None:
        touched: list[str] = []
        _forbid(monkeypatch, JevSufficiencyJudge,
                ["__init__", "assess", "judge", "rerank_and_judge"], touched)
        judge = _JitteryLaya()
        _run(judge, mode_a_config, monkeypatch, delay=SLOW_S)   # skip + finish later
        _join_finishers()
        _run(judge, mode_a_config, monkeypatch)               # reused
        judge2 = _JitteryLaya()
        judge2.assess = lambda *a, **k: acs.JudgeOutcome(None, acs.STATUS_UNAVAILABLE)
        failed = _run(judge2, mode_a_config, monkeypatch)
        assert failed.answer_check_status == "unavailable"
        assert touched == []

    def test_the_memo_never_serves_a_hosted_check(self) -> None:
        class _Hosted:  # weak-referenceable, unlike SimpleNamespace
            backend, top_k, ready, closed = "jev", 3, True, False

            def assess_if_idle(self, *a):
                raise AssertionError("a hosted check is never finished later")

        hosted = _Hosted()
        memo.store(hosted, "q", ["doc"], SufficiencyVerdict((0.9,), 0.5, "x"))
        assert memo.lookup(hosted, "q", ["doc"]) is None
        assert memo.finish_later(hosted, "q", ["doc"]) is None


# -- a switch of provider forgets every remembered verdict ----------------------------

class _Switchable:
    top_k, threshold, calibration_id = 3, 0.5, "laya:test"

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.ready, self.closed = True, False

    def shutdown(self) -> None:
        self.closed = True


class TestASwitchForgetsTheMemo:
    def test_switching_provider_and_back_starts_empty(self) -> None:
        first, online, second = _Switchable("laya"), _Switchable("jev"), _Switchable("laya")
        engine = SimpleNamespace(_sufficiency_judge=None)
        verdict = SufficiencyVerdict((0.9,), 0.5, "laya:test")
        judge_selection.swap_sufficiency_judge(engine, lambda: first)
        memo.store(first, "q", ["doc"], verdict)
        assert memo.lookup(engine._sufficiency_judge, "q", ["doc"]) is verdict

        judge_selection.swap_sufficiency_judge(engine, lambda: online)   # Settings: Jev
        assert first.closed
        assert first not in memo._memos, "the stopped judge's verdicts were kept"
        assert memo.lookup(first, "q", ["doc"]) is None

        judge_selection.swap_sufficiency_judge(engine, lambda: second)  # and back
        assert memo.lookup(engine._sufficiency_judge, "q", ["doc"]) is None

    def test_turning_the_check_off_forgets_it_too(self) -> None:
        judge = _Switchable("laya")
        engine = SimpleNamespace(_sufficiency_judge=None)
        judge_selection.swap_sufficiency_judge(engine, lambda: judge)
        memo.store(judge, "q", ["doc"], SufficiencyVerdict((0.9,), 0.5, "laya:test"))
        judge_selection.swap_sufficiency_judge(engine, lambda: None)     # Settings: Off
        assert judge.closed and judge not in memo._memos


# -- the real on-device judge, against a fake worker ---------------------------------

_FAKE_WORKER = r'''
import json, os, sys
log = os.environ.get("FAKE_LAYA_LOG")
for raw in sys.stdin:
    req = json.loads(raw)
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps({"cmd": req.get("cmd")}) + "\n")
    if req.get("cmd") == "quit":
        break
    if req.get("cmd") == "load":
        print(json.dumps({"ok": True}), flush=True)
        continue
    docs = req.get("documents", [])
    print(json.dumps({"ok": True, "probabilities": [0.3 for _ in docs]}), flush=True)
'''


def _wait(predicate, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def laya(tmp_path, monkeypatch):
    path = tmp_path / "fake_laya_worker.py"
    path.write_text(_FAKE_WORKER)
    log = tmp_path / "worker.log"
    monkeypatch.setenv("FAKE_LAYA_LOG", str(log))
    monkeypatch.setattr(laya_mod, "_WARMUP_BACKOFF_S", 0.01)
    judge = LayaSufficiencyJudge(python=sys.executable, worker_path=Path(path),
                                 timeout_s=2.0, start=False)
    yield judge, log
    judge.shutdown()


def _judge_requests(log: Path) -> int:
    if not log.exists():
        return 0
    return sum(1 for line in log.read_text().splitlines() if '"judge"' in line)


class TestTheOnDeviceCheckFinishesLater:
    def test_skip_then_judged_with_one_ask(self, laya, mode_a_config, monkeypatch) -> None:
        judge, log = laya
        judge.start_warmup()
        assert _wait(lambda: judge.ready)
        first = _run(judge, mode_a_config, monkeypatch, delay=SLOW_S)
        _join_finishers()
        second = _run(judge, mode_a_config, monkeypatch, delay=SLOW_S)
        assert first.answer_check_status == "skipped"
        assert second.answer_check_status == "judged"
        assert second.answer_check_detail == "reused"
        assert _judge_requests(log) == 3  # one per memory, each asked once

    def test_finishing_later_never_waits_for_a_busy_worker(self, laya) -> None:
        judge, _log = laya
        judge.start_warmup()
        assert _wait(lambda: judge.ready)
        result: list = []
        with judge._lock:  # a recall holds the worker for this whole block
            asker = threading.Thread(
                target=lambda: result.append(judge.assess_if_idle("q", ["doc"])))
            asker.start()
            asker.join(timeout=10)
            # It returned while the lock was still held, so it never waited for
            # the worker. (A stopwatch bound here failed on a loaded machine.)
            assert not asker.is_alive(), "finishing later waited for a busy worker"
        out = result[0]
        assert out.status == acs.STATUS_BUSY
        assert judge.assess_if_idle("q", ["doc"]).status == acs.STATUS_JUDGED

    def test_finishing_later_never_loads_a_cold_model(self, laya) -> None:
        judge, log = laya
        assert judge.assess_if_idle("q", ["doc"]).status == acs.STATUS_UNAVAILABLE
        assert memo.finish_later(judge, "q", ["doc"]) is None
        time.sleep(0.2)
        assert not judge.loading and not judge.ready and not log.exists()


# -- MCP: an older daemon's envelope is not made to look unjudged --------------------

def test_mcp_derives_the_new_fields_from_an_older_daemon() -> None:
    from superlocalmemory.mcp._recall_metadata import forward_recall_metadata
    out = forward_recall_metadata({"answer_check_status": "judged"})
    assert out["answer_check_ran"] is True and out["answer_check_note"] == ""
    out = forward_recall_metadata({"answer_check_status": "busy"})
    assert out["answer_check_ran"] is False and "did not run" in out["answer_check_note"]
