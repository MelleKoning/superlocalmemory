# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The on-device answer check recovers without thrashing, waits only inside
the recall's budget, and says why when it cannot answer.

Against a fake worker that speaks the real protocol, so no model is loaded:

* A load that can never succeed is tried once, not every few seconds forever.
* Transient load failures back off exponentially from one cycle to the next.
* A judgement that runs past its deadline costs that recall its verdict only:
  the model stays loaded, and its late answer is never read as the next one.
* A worker busy with another recall is waited for, within the budget.
* Weights other than the measured revision report confidence but cannot
  make a recall abstain.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval import sufficiency as mod
from superlocalmemory.retrieval.judge_recipe import JudgeDocument
from superlocalmemory.retrieval.sufficiency import LayaSufficiencyJudge
from tests.helpers.owned_python import owned_python

_MEASURED = "20aed815fc6acde75733882e7ec0e3f28aeb9717"

_FAKE_WORKER = r'''
import json, os, sys, time

mode = os.environ.get("FAKE_MODE", "ok")
delay = float(os.environ.get("FAKE_JUDGE_DELAY", "0"))
first_delay = float(os.environ.get("FAKE_FIRST_JUDGE_DELAY", "-1"))
load_delay = float(os.environ.get("FAKE_LOAD_DELAY", "0"))
log = os.environ.get("FAKE_LOG")

def note(kind):
    if log:
        with open(log, "a") as fh:
            fh.write(kind + "\n")

def say(obj, req):
    if isinstance(req, dict) and "id" in req:
        obj["id"] = req["id"]
    print(json.dumps(obj), flush=True)

note("spawn")
judged = 0
for raw in sys.stdin:
    req = json.loads(raw)
    cmd = req.get("cmd")
    if cmd == "quit":
        break
    if cmd == "load":
        note("load")
        time.sleep(load_delay)
        if mode == "fail_permanent":
            say({"ok": False, "error": "ModuleNotFoundError: No module named 'laya_mlx'",
                 "error_kind": "permanent"}, req)
        elif mode == "fail_transient":
            say({"ok": False, "error": "RuntimeError: out of memory",
                 "error_kind": "transient"}, req)
        else:
            say({"ok": True, "model": req.get("model")}, req)
        continue
    if cmd == "judge":
        judged += 1
        time.sleep(first_delay if judged == 1 and first_delay >= 0 else delay)
        docs = req["documents"]
        say({"ok": True, "probabilities": [0.9 if "answer" in d else 0.1 for d in docs]}, req)
'''


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    path = tmp_path / "fake_laya_worker.py"
    path.write_text(_FAKE_WORKER)
    log = tmp_path / "events.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr(mod, "_WARMUP_BACKOFF_S", 0.01)
    judges: list[LayaSufficiencyJudge] = []

    def build(**kw) -> LayaSufficiencyJudge:
        kw.setdefault("timeout_s", 1.5)
        kw.setdefault("start", True)
        judge = LayaSufficiencyJudge(python=kw.pop("python", str(owned_python(tmp_path))),
                                     worker_path=path, **kw)
        judges.append(judge)
        return judge

    def events(kind: str) -> int:
        if not log.exists():
            return 0
        return sum(1 for line in log.read_text().splitlines() if line == kind)

    build.events = events
    yield build
    for judge in judges:
        judge.shutdown()


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _docs(*texts: str) -> list[JudgeDocument]:
    return [JudgeDocument(t) for t in texts]


# ---------------------------------------------------------------------------
# M-1: no endless retry, no kill-and-reload on a slow answer
# ---------------------------------------------------------------------------

class TestALoadThatCanNeverSucceedIsTriedOnce:
    def test_a_permanent_failure_stops_every_further_attempt(self, fake, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_MODE", "fail_permanent")
        judge = fake()
        assert _wait(lambda: fake.events("load") >= 1 and not judge.loading)
        for _ in range(20):
            outcome = judge.assess("q", _docs("the answer"))
            assert outcome.status == acs.STATUS_UNAVAILABLE
            time.sleep(0.03)
        assert fake.events("spawn") == 1, "a load that can never succeed was retried"
        assert fake.events("load") == 1

    def test_an_interpreter_that_cannot_start_is_permanent(self, fake, tmp_path) -> None:
        judge = fake(python=str(tmp_path / "no-such-python"))
        assert _wait(lambda: not judge.loading)
        for _ in range(10):
            judge.assess("q", _docs("the answer"))
            time.sleep(0.02)
        assert judge.loading is False and judge.ready is False
        assert judge._permanent_error, "a missing interpreter was treated as transient"


class TestTransientFailuresBackOffAcrossCycles:
    def test_each_failed_cycle_waits_longer_before_the_next(self, fake, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_MODE", "fail_transient")
        monkeypatch.setattr(mod, "_COOLDOWN_BASE_S", 0.3)
        # The wait each failed cycle chose, measured from the moment it failed:
        # reading what is left when this thread next wakes would also measure
        # how late it woke, which on a loaded CI machine is over 0.1 s.
        waits: list[float] = []
        note_failure = LayaSufficiencyJudge._note_failure

        def timed_note_failure(judge_self) -> None:
            failed_at = time.monotonic()
            note_failure(judge_self)
            waits.append(judge_self._next_attempt_at - failed_at)

        monkeypatch.setattr(LayaSufficiencyJudge, "_note_failure", timed_note_failure)
        judge = fake()
        for cycle in range(3):
            assert _wait(lambda: not judge.loading and len(waits) > cycle)
            assert _wait(lambda: time.monotonic() >= judge._next_attempt_at, timeout=5)
            judge.assess("q", _docs("the answer"))  # a recall starts the next cycle
            assert _wait(lambda: judge.loading or fake.events("load") >= 3 * (cycle + 2))
        assert waits[1] > waits[0] + 0.1 and waits[2] > waits[1] + 0.2, waits

    def test_no_recall_starts_a_cycle_during_the_cooldown(self, fake, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_MODE", "fail_transient")
        monkeypatch.setattr(mod, "_COOLDOWN_BASE_S", 5.0)
        judge = fake()
        assert _wait(lambda: fake.events("load") >= mod._WARMUP_ATTEMPTS and not judge.loading)
        judge._failures = 1  # one failed cycle behind it: the next one must wait
        judge._next_attempt_at = time.monotonic() + 5.0
        loads = fake.events("load")
        for _ in range(10):
            assert judge.assess("q", _docs("the answer")).status == acs.STATUS_UNAVAILABLE
            time.sleep(0.02)
        assert fake.events("load") == loads


class TestASlowAnswerCostsOnlyThatVerdict:
    def test_a_timeout_neither_kills_nor_reloads_the_model(self, fake, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_JUDGE_DELAY", "0.5")
        judge = fake(timeout_s=0.2)
        assert _wait(lambda: judge.ready)
        for _ in range(3):
            assert judge.assess("q", _docs("the answer")).status == acs.STATUS_UNAVAILABLE
            time.sleep(0.7)
        assert judge.ready is True
        assert fake.events("spawn") == 1 and fake.events("load") == 1, \
            "one slow answer killed the worker and reloaded the model"

    def test_a_late_answer_is_never_read_as_the_next_one(self, fake, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_FIRST_JUDGE_DELAY", "0.5")
        judge = fake(timeout_s=0.2)
        assert _wait(lambda: judge.ready)
        assert judge.assess("q", _docs("noise")).verdict is None   # abandoned at 0.2 s
        time.sleep(0.6)                                            # its 0.1 reply arrives
        outcome = judge.assess("q", _docs("the answer"))
        assert outcome.status == acs.STATUS_JUDGED
        assert outcome.verdict.probabilities == (0.9,), "the stale answer was read"

    def test_a_worker_that_never_answers_is_replaced(self, fake, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_FIRST_JUDGE_DELAY", "60")
        monkeypatch.setattr(mod, "_WEDGED_S", 0.3)
        judge = fake(timeout_s=0.2)
        assert _wait(lambda: judge.ready)
        judge.assess("q", _docs("the answer"))
        time.sleep(0.4)
        assert judge.assess("q", _docs("the answer")).verdict is None
        assert _wait(lambda: fake.events("spawn") == 2 and judge.ready), \
            "a stuck worker was kept forever"


# ---------------------------------------------------------------------------
# M-3 + M-5: the recall's budget bounds every wait, and a busy worker is waited for
# ---------------------------------------------------------------------------

class TestItWaitsOnlyInsideTheRecallsBudget:
    def test_the_recalls_deadline_wins_over_the_judges_timeout(self, fake, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_JUDGE_DELAY", "1.0")
        judge = fake(timeout_s=1.5)
        assert _wait(lambda: judge.ready)
        started = time.monotonic()
        outcome = judge.assess("q", _docs("the answer"), deadline=time.monotonic() + 0.3)
        assert time.monotonic() - started < 0.6
        assert outcome.status == acs.STATUS_UNAVAILABLE

    def test_a_busy_worker_is_waited_for_and_both_recalls_are_judged(
        self, fake, monkeypatch,
    ) -> None:
        monkeypatch.setenv("FAKE_JUDGE_DELAY", "0.3")
        judge = fake(timeout_s=1.5)
        assert _wait(lambda: judge.ready)
        outcomes: list = []
        threads = [threading.Thread(target=lambda: outcomes.append(judge.assess(
            "q", _docs("the answer"), deadline=time.monotonic() + 1.5))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert [o.status for o in outcomes] == [acs.STATUS_JUDGED] * 2, \
            "the second recall got nothing because the first was being judged"

    def test_a_worker_that_stays_busy_past_the_budget_says_busy(
        self, fake, monkeypatch,
    ) -> None:
        monkeypatch.setenv("FAKE_JUDGE_DELAY", "0.8")
        judge = fake(timeout_s=1.5)
        assert _wait(lambda: judge.ready)
        first = threading.Thread(target=lambda: judge.assess("q", _docs("the answer")))
        first.start()
        time.sleep(0.1)
        started = time.monotonic()
        outcome = judge.assess("q", _docs("the answer"), deadline=time.monotonic() + 0.4)
        assert time.monotonic() - started < 0.6
        assert outcome.status == acs.STATUS_BUSY
        first.join(5)

    def test_a_loading_model_says_warming_and_is_never_waited_for(
        self, fake, monkeypatch,
    ) -> None:
        monkeypatch.setenv("FAKE_LOAD_DELAY", "2")
        judge = fake()
        assert _wait(lambda: judge.loading)
        started = time.monotonic()
        outcome = judge.assess("q", _docs("the answer"), deadline=time.monotonic() + 1.5)
        assert time.monotonic() - started < 0.1
        assert outcome.status == acs.STATUS_WARMING


# ---------------------------------------------------------------------------
# L-3: shutdown leaves no open pipe
# ---------------------------------------------------------------------------

def test_shutdown_closes_the_workers_pipes(fake) -> None:
    judge = fake()
    assert _wait(lambda: judge.ready)
    proc = judge._proc
    judge.shutdown()
    assert proc.poll() is not None
    assert proc.stdin.closed and proc.stdout.closed


# ---------------------------------------------------------------------------
# F16: only the measured weights may abstain
# ---------------------------------------------------------------------------

def _snapshot(tmp_path: Path, sha: str) -> str:
    return str(tmp_path / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / sha)


class TestOnlyTheMeasuredWeightsMayAbstain:
    def test_the_measured_revision_keeps_its_threshold(self, tmp_path) -> None:
        judge = LayaSufficiencyJudge(model=_snapshot(tmp_path, _MEASURED), start=False)
        assert judge.threshold == pytest.approx(0.6)
        assert judge.calibration_status == "measured_small_sample_not_calibrated"

    @pytest.mark.parametrize("model", [
        "aac6fef/laya-mlx",                        # whatever revision is cached
        "~/models/laya",                           # a folder that is not a snapshot
        "SNAPSHOT:0123456789abcdef0123456789abcdef01234567",
    ])
    def test_other_weights_report_confidence_but_cannot_abstain(self, tmp_path, model) -> None:
        if model.startswith("SNAPSHOT:"):
            model = _snapshot(tmp_path, model.split(":", 1)[1])
        judge = LayaSufficiencyJudge(model=model, start=False)
        assert judge.threshold == 0.0
        assert judge.calibration_status == "not_measured_cannot_abstain"

