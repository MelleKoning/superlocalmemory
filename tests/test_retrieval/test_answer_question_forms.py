# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A permission question the on-device answer check used to abstain on.

LAYA-A2 ("May the agent publish without approval?" against "Never publish until
the owner explicitly approves the release") was a false abstention on the pinned
weights. The fix is an explicit rule for generic-subject permission questions —
never a lower threshold, never an invented probability. LAYA-A1 (an order
question) is NOT fixed: a reworded second question fixed it on the tuning set
but added false accepts on a blind held-out set, so it was not shipped; a test
pins that order questions are judged exactly as before. A fake worker speaks the
real protocol; no model loads.
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

# Low probabilities for every question, as the real model gave on A1 and A2.
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
    if mode == "malformed":
        print(json.dumps({"ok": True, "probabilities": [7.0] * len(docs), "id": rid}), flush=True)
        continue
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


class TestWhichQuestionsTheRuleReads:
    def test_a_generic_subject_permission_question_is_read(self) -> None:
        assert forms.permission_action(A2_Q) == "publish without approval"
        assert forms.permission_action("can anyone push straight to main") == \
            "push straight to main"
        assert forms.permission_action("Is it okay to restart the server?") == \
            "restart the server"

    @pytest.mark.parametrize("question", [
        A1_Q, "what port does the api listen on", "Who owns the billing dashboard?",
        "May Rosa merge into the release branch?",  # a named subject: left to the model
        "", "x" * 400,
    ])
    def test_every_other_question_is_left_to_the_model(self, question) -> None:
        assert forms.applies(question) is False
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

    @pytest.mark.parametrize("memory", [
        "Should we never publish without approval?",                  # a question, '?'
        "Should we never publish without approval",                   # no '?', still a question
        "Open point: can we never publish without approval? Ask Rosa.",
        "Do we never publish without approval",
        "Is it true that we never publish without approval?",
    ])
    def test_a_sentence_that_is_itself_a_question_is_not_a_rule(self, memory) -> None:
        assert forms.rule_supports(A2_Q, memory) is False

    @pytest.mark.parametrize("memory", [
        "Do not publish without approval.",            # imperative "Do not", not a question
        "Never publish without approval! Ask first?",  # the rule sentence ends in '!'
        "Must not publish without approval",
    ])
    def test_an_imperative_rule_is_still_a_rule(self, memory) -> None:
        assert forms.rule_supports(A2_Q, memory) is True

    def test_a_named_entity_must_be_in_the_same_sentence(self) -> None:
        q = "Can we deploy Juniper on a Friday?"
        assert forms.rule_supports(q, "Never deploy Juniper on a Friday.") is True
        assert forms.rule_supports(q, "Heron is never deployed on Fridays.") is False
        assert forms.rule_supports(q, "Juniper is great. Never deploy on a Friday.") is False


class TestTheJudgeAppliesTheRule:
    def test_a2_is_accepted_by_the_rule_and_keeps_the_models_numbers(
            self, worker, tmp_path) -> None:
        judge = _judge(worker, tmp_path)
        try:
            verdict = judge.judge(A2_Q, A2_DOCS)
        finally:
            judge.shutdown()
        assert verdict.probabilities == (0.08, 0.02, 0.02)  # never invented
        assert verdict.answer_confidence < verdict.threshold == 0.6
        assert verdict.rule_support == (0,)
        assert verdict.insufficient is False
        assert verdict.calibration_id == judge.calibration_id + "+" + forms.FORMS_ID
        assert _queries(tmp_path)[-1:] == [A2_Q]  # asked once, as typed

    @pytest.mark.parametrize("question,docs", [
        (A1_Q, A1_DOCS),  # order: not fixed, judged exactly as before
        (A2_Q, A2_DOCS[1:]),  # the rule read it and settled nothing: the base id
        ("which team owns the billing dashboard", A2_DOCS[1:]),
    ])
    def test_every_other_question_is_one_request_and_an_unchanged_verdict(
            self, worker, tmp_path, question, docs) -> None:
        judge = _judge(worker, tmp_path)
        try:
            verdict = judge.judge(question, docs)
        finally:
            judge.shutdown()
        assert [q for q in _queries(tmp_path) if q != "warm-up"] == [question]
        assert verdict.calibration_id == judge.calibration_id
        assert verdict.rule_support == ()
        assert verdict.insufficient is True

    def test_a_malformed_answer_is_still_no_verdict_even_when_a_rule_matches(
            self, worker, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_LAYA_MODE", "malformed")
        judge = _judge(worker, tmp_path)
        try:
            outcome = judge.assess(A2_Q, A2_DOCS)
        finally:
            judge.shutdown()
        assert outcome.verdict is None  # the rule never answers without the model


class TestTheVerdictTravelsIntact:
    def test_a_rule_supported_verdict_is_not_an_abstention(self) -> None:
        verdict = SufficiencyVerdict((0.08, 0.06, 0.02), 0.6, "laya:x+permission-rule-v1",
                                     rule_support=(0,))
        response = RecallResponse(results=[RetrievalResult(
            fact=AtomicFact(content=A2_DOCS[0]), score=0.5, confidence=1.0)])
        finalize_score_contract(response, verdict)
        assert response.abstained is False
        assert response.answer_confidence == 0.08
        assert response.calibration_id == "laya:x+permission-rule-v1"

    def test_one_memory_at_a_time_combines_to_the_same_verdict(self) -> None:
        from superlocalmemory.core.answer_check_deferred import _combine

        parts = [SufficiencyVerdict((p,), 0.6, "laya:x+permission-rule-v1", rule_support=rs)
                 for p, rs in ((0.08, (0,)), (0.06, ()), (0.02, ()))]
        combined = _combine(parts)
        assert combined == SufficiencyVerdict((0.08, 0.06, 0.02), 0.6, "laya:x+permission-rule-v1",
                                              rule_support=(0,))
        assert combined.insufficient is False


@pytest.mark.native
@pytest.mark.skipif(
    not (Path(os.environ.get("SLM_TEST_LAYA_PYTHON", "/nonexistent")).exists()
         and Path(os.environ.get("SLM_TEST_LAYA_HF_HOME", "/nonexistent")).is_dir()),
    reason="set SLM_TEST_LAYA_PYTHON and SLM_TEST_LAYA_HF_HOME to a local Laya install")
def test_a2_is_fixed_and_a1_unchanged_on_the_pinned_weights(tmp_path, monkeypatch) -> None:
    """The real model: A2 accepted by the rule, the threshold still 0.6, and A1
    judged exactly as before (still a false abstention: not fixed in 4.1.22)."""
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
    assert a1.probabilities[0] == pytest.approx(0.3753, abs=0.002) and a1.insufficient is True
    assert a2.probabilities[0] == pytest.approx(0.0838, abs=0.002)
    assert a2.insufficient is False and a2.rule_support == (0,)


class TestARuleAboutSomethingElseIsNotTheAnswer:
    """A rule whose action has its own object answers only a question about that object:
    "Never publish Atlas without approval" governs Atlas, not whatever "the agent"
    may publish."""

    @pytest.mark.parametrize("memory", [
        "Never publish Atlas without approval.",
        "Never publish the billing dashboard without approval.",
        "Do not publish release notes without approval.",
    ])
    def test_a_rule_about_a_named_thing_does_not_answer_the_general_question(self, memory) -> None:
        assert forms.rule_supports(A2_Q, memory) is False

    @pytest.mark.parametrize("memory", [
        "Never publish until the owner explicitly approves the release.",
        "Never publish anything without approval.",
        "Do not publish without the owner's approval.",
    ])
    def test_a_rule_about_the_asked_action_itself_still_answers(self, memory) -> None:
        assert forms.rule_supports(A2_Q, memory) is True

    def test_the_named_thing_answers_when_the_question_names_it(self) -> None:
        q = "May the agent publish Atlas without approval?"
        assert forms.rule_supports(q, "Never publish Atlas without approval.") is True


def test_a_passive_rule_has_no_object_to_narrow_it() -> None:
    q = "Can we delete the Vault9 audit logs?"
    assert forms.rule_supports(q, "Vault9 audit logs must never be deleted, archived only.") is True


def test_one_memory_at_a_time_keeps_a_mix_of_rule_and_model_verdicts() -> None:
    """A rule-settled part carries the suffixed id and a
    model-only part the base id; combining them must not drop the verdict."""
    from superlocalmemory.core.answer_check_deferred import _combine
    from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict

    base = "laya:test@x:sufficiency-v1:top3"
    ruled = SufficiencyVerdict((0.08,), 0.6, forms.calibration_id_for(base, (0,)),
                               "calibrated", "laya", (0,))
    plain = SufficiencyVerdict((0.06,), 0.6, base, "calibrated", "laya", ())
    combined = _combine([ruled, plain])
    assert combined is not None
    assert combined.rule_support == (0,)
    assert combined.calibration_id == forms.calibration_id_for(base, (0,))
    assert combined.probabilities == (0.08, 0.06)
    assert _combine([plain, plain]).calibration_id == base
