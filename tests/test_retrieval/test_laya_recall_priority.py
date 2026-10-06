# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Typing borrows the answer check's Laya worker and must never cost a recall its verdict.

The worker serves one request at a time. Typing therefore runs only on
background threads, only when no recall is in flight, in small chunks that
give the worker back between them, and never starts (or re-warms) a model of
its own. A recall that arrives while a typing chunk holds the worker waits for
it inside its own deadline instead of going unjudged.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.core import recall_gate
from superlocalmemory.encoding.memory_kind_recipe import KINDS_V1, KindAnswer
from superlocalmemory.retrieval import sufficiency as mod
from superlocalmemory.retrieval.sufficiency import LayaSufficiencyJudge, SufficiencyVerdict
from tests.helpers.owned_python import owned_python

_FAKE_WORKER = r'''
import json, os, sys, time

mode = os.environ.get("FAKE_LAYA_MODE", "normal")
log = os.environ.get("FAKE_LAYA_LOG")
for raw in sys.stdin:
    req = json.loads(raw)
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps(req) + "\n")
    cmd = req.get("cmd")
    rid = req.get("id")
    def say(payload):
        if rid is not None:
            payload["id"] = rid
        print(json.dumps(payload), flush=True)
    if cmd == "quit":
        break
    if cmd == "load":
        say({"ok": True, "model": req.get("model")})
        continue
    if cmd == "judge":
        say({"ok": True, "probabilities": [0.9 if "answer" in d else 0.1
                                           for d in req["documents"]]})
        continue
    if cmd == "kinds":
        time.sleep(0.12)
        labels = list(req["criteria"])
        verify = set((req.get("verify") or {}).get("indices", []))
        choice = "nonsense" if mode == "kinds_bad" else labels[1]
        answers = [{"choice": choice,
                    "probabilities": {l: (0.5 if l == labels[1] else 0.05) for l in labels},
                    "confidence": 0.4, "verify": (0.9 if i in verify else None)}
                   for i, _ in enumerate(req["documents"])]
        if mode == "kinds_short":
            answers = answers[:-1]
        say({"ok": True, "answers": answers})
        continue
    say({"ok": False, "error": "unknown"})
'''


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    path = tmp_path / "fake_laya_worker.py"
    path.write_text(_FAKE_WORKER, encoding="utf-8")
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_LAYA_LOG", str(log))
    monkeypatch.setattr(mod, "_WARMUP_BACKOFF_S", 0.01)
    return path, log


def _judge(path: Path, **kw) -> LayaSufficiencyJudge:
    kw.setdefault("timeout_s", 2.0)
    return LayaSufficiencyJudge(python=str(owned_python(path.parent)), worker_path=path, **kw)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _requests(log: Path, cmd: str) -> list[dict]:
    if not log.exists():
        return []
    return [r for r in map(json.loads, log.read_text(encoding="utf-8").splitlines()) if r.get("cmd") == cmd]


def _in_background(fn):
    """Run ``fn`` on a background-work thread; returns (thread, result box)."""
    box: dict = {}

    def run():
        with recall_gate.background_work():
            box["value"] = fn()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


@pytest.fixture()
def warm(worker):
    path, log = worker
    judge = _judge(path)
    assert _wait(lambda: judge.ready)
    yield judge, log
    judge.shutdown()


def test_recall_waits_for_background_holder(warm) -> None:
    judge, log = warm
    thread, box = _in_background(
        lambda: judge.ask_kinds(["first memory", "second memory"], KINDS_V1, []))
    assert _wait(lambda: len(_requests(log, "kinds")) == 1, 5.0)
    started = time.monotonic()
    verdict = judge.judge("q", ["the answer"])      # worker busy for ~120 ms more
    waited = time.monotonic() - started
    thread.join(5)
    assert isinstance(verdict, SufficiencyVerdict), "the recall went unjudged"
    assert verdict.probabilities == (0.9,)
    assert waited < 1.0
    assert box["value"] is not None and len(box["value"]) == 2


def test_ask_kinds_refuses_on_foreground_thread(warm) -> None:
    judge, log = warm
    assert judge.ask_kinds(["a memory"], KINDS_V1, []) is None
    assert _requests(log, "kinds") == []


def test_ask_kinds_waits_for_foreground_idle(warm, monkeypatch) -> None:
    judge, log = warm
    # The old version of this test slept 0.3s and trusted that ask_kinds
    # would have reached (and blocked on) wait_for_foreground_idle by then -
    # a wall-clock margin that a loaded machine can blow past before the
    # background thread is even scheduled, with nothing to show it ever
    # entered the wait it's supposed to be testing. Instrument the real
    # call instead: the Event fires exactly when ask_kinds reaches it, no
    # matter how long scheduling takes to get there.
    entered_wait = threading.Event()
    real_wait_for_foreground_idle = recall_gate.wait_for_foreground_idle

    def _observed_wait_for_foreground_idle() -> None:
        entered_wait.set()
        real_wait_for_foreground_idle()

    monkeypatch.setattr(recall_gate, "wait_for_foreground_idle",
                        _observed_wait_for_foreground_idle)

    recall_gate.begin_recall()
    try:
        thread, box = _in_background(lambda: judge.ask_kinds(["a memory"], KINDS_V1, []))
        # begin_recall() above already ran before the thread was started, so
        # once ask_kinds reaches the real wait it is guaranteed to block -
        # this is a hang-safety bound, not a guess at how long work takes.
        assert entered_wait.wait(5), "ask_kinds never reached the foreground-idle wait"
        assert _requests(log, "kinds") == [], "typing ran while a recall was in flight"
    finally:
        recall_gate.end_recall()
    assert _wait(lambda: not thread.is_alive(), 5.0), \
        "ask_kinds never returned after the recall ended"
    assert len(_requests(log, "kinds")) == 1
    answers = box["value"]
    assert answers == [KindAnswer(choice=list(KINDS_V1.criteria)[1],
                                  probabilities=answers[0].probabilities,
                                  confidence=0.4, verify=None)]


def test_background_gives_the_worker_back_between_chunks(warm) -> None:
    judge, log = warm
    docs = [f"memory {i}" for i in range(5)]
    thread, box = _in_background(lambda: judge.ask_kinds(docs, KINDS_V1, [0, 3]))
    thread.join(10)
    sent = _requests(log, "kinds")
    assert [len(r["documents"]) for r in sent] == [2, 2, 1]
    assert [r.get("verify", {}).get("indices") for r in sent] == [[0], [1], None]
    assert [a.verify for a in box["value"]] == [0.9, None, None, 0.9, None]
    assert all(r["instructions"] == KINDS_V1.instructions for r in sent)
    assert all(r["criteria"] == dict(KINDS_V1.criteria) for r in sent)


def test_ask_kinds_never_starts_a_cold_worker(worker) -> None:
    path, log = worker
    judge = _judge(path, start=False)
    try:
        thread, box = _in_background(lambda: judge.ask_kinds(["a memory"], KINDS_V1, []))
        thread.join(5)
        assert box["value"] is None
        time.sleep(0.2)
        assert not judge.loading and not judge.ready
        assert _requests(log, "load") == [], "typing started a model"
    finally:
        judge.shutdown()


@pytest.mark.parametrize("mode", ["kinds_bad", "kinds_short"])
def test_a_malformed_kinds_answer_is_never_trusted(worker, monkeypatch, mode) -> None:
    monkeypatch.setenv("FAKE_LAYA_MODE", mode)
    path, _ = worker
    judge = _judge(path)
    try:
        assert _wait(lambda: judge.ready)
        thread, box = _in_background(lambda: judge.ask_kinds(["a", "b"], KINDS_V1, []))
        thread.join(5)
        assert box["value"] is None
        # The worker is still fine for the next recall.
        assert judge.judge("q", ["the answer"]) is not None
    finally:
        judge.shutdown()


def test_nothing_to_type_sends_nothing(warm) -> None:
    judge, log = warm
    thread, box = _in_background(lambda: judge.ask_kinds([], KINDS_V1, []))
    thread.join(5)
    assert box["value"] == []
    assert _requests(log, "kinds") == []
