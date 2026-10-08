# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Order and permission questions the on-device answer check used to abstain on.

LAYA-A1 ("Which mobile platform was chosen first?" against "We decided to ship
on iOS first and Android one quarter later") and LAYA-A2 ("May the agent publish
without approval?" against "Never publish until the owner explicitly approves the
release") were false abstentions on the pinned weights. The fix is a reworded
second question for those shapes only and an explicit rule for generic-subject
permission questions — never a lower threshold. These tests pin both parts, that
every other question is asked exactly as before, and that the verdict names the
new calibration identity. A fake worker speaks the real protocol; no model loads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from superlocalmemory.core.score_contract import finalize_score_contract
from superlocalmemory.retrieval import answer_question_forms as forms
from superlocalmemory.retrieval import sufficiency as mod
from superlocalmemory.retrieval.sufficiency import LayaSufficiencyJudge, SufficiencyVerdict
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult
from tests.helpers.owned_python import owned_python

A1_Q = "Which mobile platform was chosen first?"
A1_DOCS = ["We decided to ship Kestrel on iOS first and Android one quarter later.",
           "Beta has 40 technicians.", "Design review is Tuesday."]
A2_Q = "May the agent publish without approval?"
A2_DOCS = ["Never publish until the owner explicitly approves the release.",
           "The team uses Rust.", "Tests run each Friday."]
_SNAPSHOT_SHA = "20aed815fc6acde75733882e7ec0e3f28aeb9717"

# Low for every original question; the order rewording lifts memories that say
# "first"; a worker that breaks on the rewording only (mode "reword_fails").
_FAKE_WORKER = r'''
import json, os, sys
log = os.environ.get("FAKE_LAYA_LOG")
mode = os.environ.get("FAKE_LAYA_MODE", "normal")
for raw in sys.stdin:
    req = json.loads(raw)
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps(req) + "\n")
    rid = req.get("id")
    cmd = req.get("cmd")
    if cmd == "quit":
        break
    if cmd == "load":
        print(json.dumps({"ok": True, "id": rid}), flush=True)
        continue
    q = req["query"]
    docs = req["documents"]
    reworded = q.startswith("What is the order") or q.startswith("Is it allowed")
    if reworded and mode == "reword_fails":
        print(json.dumps({"ok": False, "error": "boom", "id": rid}), flush=True)
        continue
    if q.startswith("What is the order"):
        probs = [0.62 if "first" in d else 0.05 for d in docs]
    else:
        probs = [0.38 if "first" in d else (0.08 if "Never" in d else 0.02) for d in docs]
    print(json.dumps({"ok": True, "probabilities": probs, "id": rid}), flush=True)
'''


@pytest.fixture()
def worker(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "fake_forms_worker.py"
    path.write_text(_FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(mod, "_WARMUP_BACKOFF_S", 0.01)
    monkeypatch.setenv("FAKE_LAYA_LOG", str(tmp_path / "requests.log"))
    return path


def _judge(worker: Path, tmp_path: Path) -> LayaSufficiencyJudge:
    model = tmp_path / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / _SNAPSHOT_SHA
    judge = LayaSufficiencyJudge(python=str(owned_python(worker.parent)), worker_path=worker,
                                 model=str(model), timeout_s=3.0)
    deadline = time.monotonic() + 10
    while not judge.ready and time.monotonic() < deadline:
        time.sleep(0.02)
    assert judge.ready
    return judge


def _queries(tmp_path: Path) -> list[str]:
    log = tmp_path / "requests.log"
    return [json.loads(line)["query"] for line in log.read_text().splitlines()
            if json.loads(line).get("cmd") == "judge"]


class TestWhichQuestionsAreReworded:
    def test_the_two_regressions_get_their_rewording(self) -> None:
        assert forms.reworded(A1_Q) == ("What is the order or sequence: " + A1_Q,)
        assert forms.reworded(A2_Q) == ("Is it allowed to publish without approval?",)

    @pytest.mark.parametrize("question", [
        "what port does the api listen on", "Who owns the billing dashboard?",
        "May Rosa merge into the release branch?",  # a named subject: left to the model
        "", "x" * 400,
    ])
    def test_every_other_question_is_asked_as_typed(self, question) -> None:
        assert forms.reworded(question) == ()
        assert forms.rule_support(question, A2_DOCS) == ()


class TestThePermissionRule:
    def test_a2_is_recognised_on_its_answer_bearing_memory_only(self) -> None:
        assert forms.rule_support(A2_Q, A2_DOCS) == (0,)

    @pytest.mark.parametrize("memory", [
        "Never publish on Fridays.",                                  # approval not covered
        "We never discussed whether to publish without approval.",    # cue governs another verb
        "The agent published the release notes after approval.",     # no rule at all
        "Never deploy without approval.",                             # another action
    ])
    def test_near_misses_are_not_recognised(self, memory) -> None:
        assert forms.rule_supports(A2_Q, memory) is False

    def test_a_named_entity_must_be_in_the_same_sentence(self) -> None:
        q = "Can we deploy Juniper on a Friday?"
        assert forms.rule_supports(q, "Never deploy Juniper on a Friday.") is True
        assert forms.rule_supports(q, "Heron is never deployed on Fridays.") is False
        assert forms.rule_supports(q, "Juniper is great. Never deploy on a Friday.") is False


class TestTheJudgeAsksBothWordingsAndKeepsTheHigher:
    def test_a1_is_accepted_through_the_order_rewording(self, worker, tmp_path) -> None:
        judge = _judge(worker, tmp_path)
        try:
            verdict = judge.judge(A1_Q, A1_DOCS)
        finally:
            judge.shutdown()
        assert verdict.probabilities == (0.62, 0.05, 0.05)
        assert verdict.insufficient is False
        assert verdict.calibration_id == judge.calibration_id + "+" + forms.FORMS_ID
        assert _queries(tmp_path)[-2:] == [A1_Q, "What is the order or sequence: " + A1_Q]

    def test_a2_is_accepted_by_the_rule_and_keeps_the_models_numbers(
            self, worker, tmp_path) -> None:
        judge = _judge(worker, tmp_path)
        try:
            verdict = judge.judge(A2_Q, A2_DOCS)
        finally:
            judge.shutdown()
        assert verdict.answer_confidence == pytest.approx(0.08)  # never invented
        assert verdict.answer_confidence < verdict.threshold == 0.6
        assert verdict.rule_support == (0,)
        assert verdict.insufficient is False
        assert verdict.calibration_id.endswith("+" + forms.FORMS_ID)

    def test_a_plain_question_is_one_request_and_an_unchanged_verdict(
            self, worker, tmp_path) -> None:
        judge = _judge(worker, tmp_path)
        try:
            verdict = judge.judge("which team owns the billing dashboard", A2_DOCS[1:])
        finally:
            judge.shutdown()
        assert _queries(tmp_path)[-1:] == ["which team owns the billing dashboard"]
        assert len([q for q in _queries(tmp_path) if q != "warm-up"]) == 1
        assert verdict.calibration_id == judge.calibration_id
        assert verdict.rule_support == ()
        assert verdict.insufficient is True

    def test_a_failed_rewording_costs_the_whole_verdict_never_half(
            self, worker, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_LAYA_MODE", "reword_fails")
        judge = _judge(worker, tmp_path)
        try:
            outcome = judge.assess(A1_Q, A1_DOCS)
        finally:
            judge.shutdown()
        assert outcome.verdict is None


class TestTheVerdictTravelsIntact:
    def test_a_rule_supported_verdict_is_not_an_abstention(self) -> None:
        verdict = SufficiencyVerdict((0.08, 0.06, 0.02), 0.6, "laya:x+qforms-v1",
                                     rule_support=(0,))
        response = RecallResponse(results=[RetrievalResult(
            fact=AtomicFact(content=A2_DOCS[0]), score=0.5, confidence=1.0)])
        finalize_score_contract(response, verdict)
        assert response.abstained is False
        assert response.answer_confidence == 0.08
        assert response.calibration_id == "laya:x+qforms-v1"

    def test_one_memory_at_a_time_combines_to_the_same_verdict(self) -> None:
        from superlocalmemory.core.answer_check_deferred import _combine

        parts = [SufficiencyVerdict((p,), 0.6, "laya:x+qforms-v1", rule_support=rs)
                 for p, rs in ((0.08, (0,)), (0.06, ()), (0.02, ()))]
        combined = _combine(parts)
        assert combined == SufficiencyVerdict((0.08, 0.06, 0.02), 0.6, "laya:x+qforms-v1",
                                              rule_support=(0,))
        assert combined.insufficient is False


@pytest.mark.native
@pytest.mark.skipif(
    not (Path(os.environ.get("SLM_TEST_LAYA_PYTHON", "/nonexistent")).exists()
         and Path(os.environ.get("SLM_TEST_LAYA_HF_HOME", "/nonexistent")).is_dir()),
    reason="set SLM_TEST_LAYA_PYTHON and SLM_TEST_LAYA_HF_HOME to a local Laya install")
def test_a1_and_a2_on_the_pinned_weights(tmp_path, monkeypatch) -> None:
    """The real model: both regressions accepted, the threshold still 0.6."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    hf_home = os.environ["SLM_TEST_LAYA_HF_HOME"]
    model = Path(hf_home) / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / _SNAPSHOT_SHA
    judge = LayaSufficiencyJudge(python=os.environ["SLM_TEST_LAYA_PYTHON"], model=str(model),
                                 hf_home=hf_home, timeout_s=10.0)
    try:
        deadline = time.monotonic() + 240
        while not judge.ready and time.monotonic() < deadline:
            time.sleep(0.2)
        a1, a2 = judge.judge(A1_Q, A1_DOCS), judge.judge(A2_Q, A2_DOCS)
    finally:
        judge.shutdown()
    assert a1.threshold == a2.threshold == 0.6
    assert a1.insufficient is False and a1.probabilities[0] >= 0.6
    assert a2.insufficient is False and a2.rule_support == (0,)
