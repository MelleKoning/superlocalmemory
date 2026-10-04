# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A check finished after its recall returned never costs a live recall its verdict.

Seen in the 4.1.20 audit: a recall that used its budget queued its check to be
finished later; that check held the on-device worker for a whole judgement, so
a different question asked 50 ms later came back "busy", and the same question
asked again came back "unavailable" — the work meant to make the next recall
judged made it unjudged. Each test here runs the real on-device judge against a
fake worker that takes ``DOC_S`` per memory, as the real one does.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core import answer_check_deferred as deferred
from superlocalmemory.core import answer_check_memo as memo
from superlocalmemory.core import recall_gate
from superlocalmemory.core.answer_check_stage import run_answer_check
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval import sufficiency as laya_mod
from superlocalmemory.retrieval.sufficiency import LayaSufficiencyJudge, SufficiencyVerdict

#: Seconds the fake worker spends on each memory.
DOC_S = 0.3
#: What a live recall has left of its budget when its check starts. One whole
#: judgement (3 x DOC_S) fits, with room for one memory's wait in front of it,
#: but not for a whole judgement's wait (the defect).
LEFT_S = 1.5

_WORKER = r'''
import json, os, sys, time
log = os.environ["FAKE_LAYA_LOG"]
per_doc = float(os.environ["FAKE_LAYA_DOC_S"])
def note(**kw):
    with open(log, "a") as fh:
        fh.write(json.dumps(kw) + "\n")
for raw in sys.stdin:
    req = json.loads(raw)
    cmd = req.get("cmd")
    if cmd == "quit":
        break
    if cmd == "load":
        print(json.dumps({"id": req.get("id"), "ok": True}), flush=True)
        continue
    docs = req.get("documents", [])
    note(event="start", query=req.get("query"), docs=len(docs), at=time.monotonic())
    probs = []
    for d in docs:
        time.sleep(per_doc)
        probs.append(round(0.05 + (sum(map(ord, d)) % 90) / 100, 4))
    note(event="end", query=req.get("query"), docs=len(docs), at=time.monotonic())
    print(json.dumps({"id": req.get("id"), "ok": True, "probabilities": probs}), flush=True)
'''


def _wait(predicate, timeout: float = 15.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _events(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _starts(log: Path, query: str | None = None) -> list[dict]:
    return [e for e in _events(log) if e["event"] == "start"
            and (query is None or e["query"] == query)]


@pytest.fixture
def laya(tmp_path, monkeypatch):
    worker = tmp_path / "fake_laya_worker.py"
    worker.write_text(_WORKER)
    log = tmp_path / "worker.log"
    monkeypatch.setenv("FAKE_LAYA_LOG", str(log))
    monkeypatch.setenv("FAKE_LAYA_DOC_S", str(DOC_S))
    monkeypatch.setattr(laya_mod, "_WARMUP_BACKOFF_S", 0.01)
    judge = LayaSufficiencyJudge(python=sys.executable, worker_path=worker,
                                 timeout_s=5.0, start=False)
    judge.start_warmup()
    assert _wait(lambda: judge.ready), "fake worker did not load"
    yield judge, log
    judge.shutdown()
    _join_finishers()


def _join_finishers(timeout: float = 15.0) -> None:
    for thread in threading.enumerate():
        if thread.name == deferred.THREAD_NAME:
            thread.join(timeout)


def _response(*texts: str) -> SimpleNamespace:
    return SimpleNamespace(results=[
        SimpleNamespace(fact=SimpleNamespace(content=t, profile_id="default")) for t in texts])


ALPHA = ("alpha fact one", "alpha fact two", "alpha fact three")
BETA = ("beta fact one", "beta fact two", "beta fact three")


def _documents(judge, texts) -> list:
    from superlocalmemory.core.answer_check_stage import _top_documents
    return _top_documents(judge, _response(*texts))


def _live_recall(judge, query: str, texts, left_s: float = LEFT_S) -> acs.JudgeOutcome:
    """A recall whose retrieval left ``left_s`` of the budget for its check."""
    started = (time.monotonic() - acs.RECALL_CEILING_S + acs.POST_JUDGE_RESERVE_S + left_s)
    engine = SimpleNamespace(_sufficiency_judge=judge)
    return run_answer_check(engine, query, _response(*texts), recall_started=started)


def _start_deferred(judge, log: Path, query: str, texts) -> threading.Thread:
    thread = memo.finish_later(judge, query, _documents(judge, texts))
    assert thread is not None
    assert _wait(lambda: _starts(log, query)), "the deferred check never started"
    return thread


class TestALiveRecallComesFirst:
    def test_a_new_question_during_a_deferred_check_is_judged(self, laya) -> None:
        judge, log = laya
        _start_deferred(judge, log, "what is alpha?", ALPHA)
        out = _live_recall(judge, "what is beta?", BETA)
        assert out.status == acs.STATUS_JUDGED, out
        assert out.detail == acs.DETAIL_NONE
        # The deferred check carried on afterwards and finished.
        _join_finishers()
        assert memo.lookup(judge, "what is alpha?", _documents(judge, ALPHA)) is not None

    def test_the_same_question_during_a_deferred_check_is_judged(self, laya) -> None:
        judge, log = laya
        _start_deferred(judge, log, "what is alpha?", ALPHA)
        out = _live_recall(judge, "what is alpha?", ALPHA)
        assert out.status == acs.STATUS_JUDGED, out

    def test_a_recall_that_arrives_mid_check_finds_the_worker_free(self, laya) -> None:
        """The daemon's flow: the recall is in flight from its first line
        (``recall_gate``), so the check finished later stops before its next
        memory, while the recall is still retrieving. Its budget here is one
        whole judgement plus a little: no room to wait for anything."""
        judge, log = laya
        _start_deferred(judge, log, "what is alpha?", ALPHA)
        recall_gate.begin_recall()
        try:
            time.sleep(DOC_S + 0.05)  # retrieval
            out = _live_recall(judge, "what is beta?", BETA, left_s=3 * DOC_S + 0.35)
        finally:
            recall_gate.end_recall()
        assert out.status == acs.STATUS_JUDGED, out

    def test_nothing_is_asked_while_a_recall_is_in_flight(self, laya) -> None:
        judge, log = laya
        recall_gate.begin_recall()
        try:
            thread = memo.finish_later(judge, "what is alpha?", _documents(judge, ALPHA))
            assert thread is not None
            time.sleep(4 * deferred.POLL_S + DOC_S)
            assert _starts(log) == [], "a deferred check ran during a live recall"
        finally:
            recall_gate.end_recall()
        _join_finishers()
        # One memory per request, so a recall arriving waits for one at most.
        assert [e["docs"] for e in _starts(log)] == [1, 1, 1]
        assert memo.lookup(judge, "what is alpha?", _documents(judge, ALPHA)) is not None

    def test_one_memory_at_a_time_gives_the_same_verdict(self, laya) -> None:
        judge, _log = laya
        docs = _documents(judge, ALPHA)
        whole = judge.assess("what is alpha?", docs).verdict
        memo.finish_later(judge, "what is alpha?", docs)
        _join_finishers()
        assert memo.lookup(judge, "what is alpha?", docs) == whole


class TestAVerdictThatArrivedWhileWaitingIsReused:
    def test_reused_once_the_worker_is_free(self, laya) -> None:
        judge, log = laya
        docs = _documents(judge, ALPHA)
        known = SufficiencyVerdict((0.7, 0.2, 0.1), judge.threshold, judge.calibration_id)
        result: list = []
        with judge._lock:  # someone else has the worker
            asker = threading.Thread(target=lambda: result.append(judge.assess(
                "what is alpha?", docs, deadline=time.monotonic() + 3.0,
                reuse=lambda: memo.lookup(judge, "what is alpha?", docs))))
            asker.start()
            time.sleep(0.1)
            memo.store(judge, "what is alpha?", docs, known)  # finished meanwhile
        asker.join(5)
        assert result[0].status == acs.STATUS_JUDGED
        assert result[0].detail == acs.DETAIL_REUSED and result[0].verdict is known
        assert _starts(log) == []  # nothing was sent to the model


# -- the queue, against fakes (no worker) -----------------------------------------

class _Judge:
    backend, top_k, threshold, calibration_id = "laya", 3, 0.5, "laya:test"

    def __init__(self) -> None:
        self.ready, self.closed = True, False
        self.asked: list[str] = []
        self.on_ask = None

    def assess_if_idle(self, query, documents):
        self.asked.append(documents[0])
        if self.on_ask:
            self.on_ask()
        verdict = SufficiencyVerdict((0.4,), self.threshold, self.calibration_id)
        return acs.JudgeOutcome(verdict, acs.STATUS_JUDGED)


class TestTheQueue:
    def test_an_interrupted_check_resumes_where_it_stopped(self, monkeypatch) -> None:
        monkeypatch.setattr(deferred, "WAIT_FOR_IDLE_S", 0.2)
        judge = _Judge()
        hold = memo.live_check(judge)
        judge.on_ask = lambda: (hold.__enter__(), setattr(judge, "on_ask", None))
        memo.finish_later(judge, "q", ["a", "b", "c"])
        _join_finishers()
        assert judge.asked == ["a"]           # a recall arrived; it stopped
        assert memo.lookup(judge, "q", ["a", "b", "c"]) is None
        hold.__exit__(None, None, None)
        memo.finish_later(judge, "q", ["a", "b", "c"])
        _join_finishers()
        assert judge.asked == ["a", "b", "c"]  # carried on; "a" not asked twice
        assert memo.lookup(judge, "q", ["a", "b", "c"]).probabilities == (0.4, 0.4, 0.4)

    def test_every_queued_question_is_finished(self) -> None:
        judge = _Judge()
        gate = threading.Event()
        judge.on_ask = gate.wait
        memo.finish_later(judge, "q1", ["a"])
        memo.finish_later(judge, "q2", ["b"])  # queued behind q1, not dropped
        gate.set()
        _join_finishers()
        assert memo.lookup(judge, "q1", ["a"]) is not None
        assert memo.lookup(judge, "q2", ["b"]) is not None

    def test_a_stopped_judge_answers_nothing(self) -> None:
        judge = _Judge()
        memo.store(judge, "q", ["a"], SufficiencyVerdict((0.4,), 0.5, "laya:test"))
        judge.closed = True
        assert memo.lookup(judge, "q", ["a"]) is None
        assert memo.finish_later(judge, "q", ["a"]) is None
