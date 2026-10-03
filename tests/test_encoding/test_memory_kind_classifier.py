# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The kind classifier: caller first, then one model (only one already running), then rules.

Every judge here is a fake. The classifier must never build one of its own
(LLD I5), never ask Jev without the separate, literal-True typing consent, and
never fail: any trouble falls back to the rules suggestion for each fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from superlocalmemory.encoding.memory_kind_classifier import (
    MAX_MODEL_FACTS,
    KindBackend,
    KindClassifier,
    resolve_kind_backend,
    with_kind,
)
from superlocalmemory.encoding.memory_kind_recipe import KINDS_V1, KindAnswer
from superlocalmemory.encoding.memory_kind_rules import RULES_RECIPE
from superlocalmemory.storage.memory_kinds import (
    KindAssignment,
    KindSource,
    MemoryKind,
)
from superlocalmemory.storage.models import AtomicFact, FactType, Mode

K = MemoryKind


def _cfg(**kw: Any) -> SimpleNamespace:
    base = {"enabled": True, "backend": "auto", "jev_consent": False,
            "display_min_confidence": 0.20}
    base.update(kw)
    return SimpleNamespace(**base)


def _answer(choice: str, confidence: float = 0.6, verify: float | None = None) -> KindAnswer:
    return KindAnswer(choice=choice, probabilities={choice: confidence},
                      confidence=confidence, verify=verify)


@dataclass
class _FakeJudge:
    backend: str
    ready: bool = True
    answers: Any = None            # list, None, or an exception to raise
    calls: list = field(default_factory=list)

    def ask_kinds(self, documents, recipe, verify_indices, **kw):
        self.calls.append({"documents": list(documents), "recipe": recipe,
                           "verify": list(verify_indices), **kw})
        if isinstance(self.answers, BaseException):
            raise self.answers
        if callable(self.answers):
            return self.answers(documents)
        return self.answers


def _facts(*texts: str, fact_type: FactType = FactType.SEMANTIC) -> list[AtomicFact]:
    return [AtomicFact(content=t, fact_type=fact_type) for t in texts]


def _clf(cfg=None, mode=Mode.A, judge=None, llm=False) -> KindClassifier:
    return KindClassifier(config=cfg or _cfg(), mode=mode,
                          judge_supplier=lambda: judge, llm_available=llm)


# -- source priority ----------------------------------------------------------

def test_caller_kind_applies_to_all_facts() -> None:
    judge = _FakeJudge("laya", answers=[_answer(K.EPISODIC.value)] * 3)
    out = _clf(judge=judge).suggest(_facts("a fact here", "another fact", "third one"),
                                    caller_kind=K.DECISION)
    assert [a.kind for a in out] == [K.DECISION] * 3
    assert {a.source for a in out} == {KindSource.CALLER}
    assert judge.calls == [], "a declared kind must not cost a model call"


def test_off_returns_none_for_every_fact() -> None:
    facts = _facts("Never run rm -rf here.", "We chose Postgres.")
    assert _clf(_cfg(enabled=False)).suggest(facts, caller_kind=None) == [None, None]
    assert _clf(_cfg(backend="off")).suggest(facts, caller_kind=None) == [None, None]
    # A declared kind is the caller's write, not a suggestion: it still applies.
    assert _clf(_cfg(backend="off")).suggest(facts, caller_kind=K.RULE)[0].kind is K.RULE


def test_rules_backend_never_asks_a_model() -> None:
    judge = _FakeJudge("laya", answers=[_answer(K.OPINION.value)])
    out = _clf(_cfg(backend="rules"), judge=judge).suggest(
        _facts("Never run rm -rf with a glob."), caller_kind=None)
    assert out[0].kind is K.RULE and out[0].source is KindSource.RULES
    assert judge.calls == []


# -- backend resolution (exclusivity: the one live judge, or none) ------------

def test_auto_never_picks_jev() -> None:
    jev = _FakeJudge("jev", answers=[_answer(K.OPINION.value)])
    for mode in (Mode.A, Mode.B, Mode.C):
        backend = resolve_kind_backend(_cfg(jev_consent=True), mode, jev, False)
        assert backend is KindBackend.RULES
    out = _clf(_cfg(jev_consent=True), judge=jev).suggest(_facts("plain fact text"),
                                                           caller_kind=None)
    assert out[0].source is KindSource.RULES
    assert jev.calls == []


@pytest.mark.parametrize("consent", [False, "true", "True", 1, None, "yes"])
def test_jev_needs_jev_judge_and_literal_true_consent(consent: object) -> None:
    jev = _FakeJudge("jev", answers=[_answer(K.OPINION.value)])
    assert resolve_kind_backend(_cfg(backend="jev", jev_consent=consent), Mode.A, jev,
                                False) is KindBackend.RULES
    out = _clf(_cfg(backend="jev", jev_consent=consent), judge=jev).suggest(
        _facts("plain fact text"), caller_kind=None)
    assert out[0].source is KindSource.RULES and jev.calls == []


def test_jev_with_consent_and_jev_judge_is_asked_once_with_consent() -> None:
    jev = _FakeJudge("jev", answers=[_answer(K.OPINION.value, 0.7)])
    cfg = _cfg(backend="jev", jev_consent=True)
    assert resolve_kind_backend(cfg, Mode.A, jev, False) is KindBackend.JEV
    # Jev configured, but the live judge is Laya: no Jev, and no Laya either.
    laya = _FakeJudge("laya", answers=[_answer(K.OPINION.value)])
    assert resolve_kind_backend(cfg, Mode.A, laya, False) is KindBackend.RULES
    out = _clf(cfg, judge=jev).suggest(_facts("plain fact text"), caller_kind=None)
    assert out[0] == KindAssignment(K.OPINION, KindSource.MODEL_JEV, 0.7, KINDS_V1.recipe_id)
    assert len(jev.calls) == 1 and jev.calls[0]["consent"] is True


def test_laya_needs_running_laya_judge() -> None:
    cfg = _cfg(backend="laya")
    assert resolve_kind_backend(cfg, Mode.A, None, False) is KindBackend.RULES
    assert resolve_kind_backend(cfg, Mode.A, _FakeJudge("jev"), False) is KindBackend.RULES
    assert resolve_kind_backend(cfg, Mode.A, _FakeJudge("laya"), False) is KindBackend.LAYA
    cold = _FakeJudge("laya", ready=False, answers=[_answer(K.OPINION.value)])
    out = _clf(cfg, judge=cold).suggest(_facts("plain fact text"), caller_kind=None)
    assert out[0].source is KindSource.RULES
    assert cold.calls == [], "a judge that is not running is not asked (or started)"


def test_auto_prefers_llm_in_mode_b_c_then_laya_then_rules() -> None:
    laya = _FakeJudge("laya")
    assert resolve_kind_backend(_cfg(), Mode.B, laya, True) is KindBackend.LLM
    assert resolve_kind_backend(_cfg(), Mode.C, laya, True) is KindBackend.LLM
    assert resolve_kind_backend(_cfg(), Mode.B, laya, False) is KindBackend.LAYA
    assert resolve_kind_backend(_cfg(), Mode.A, laya, True) is KindBackend.LAYA
    assert resolve_kind_backend(_cfg(), Mode.A, None, True) is KindBackend.RULES
    assert resolve_kind_backend(_cfg(backend="llm"), Mode.A, laya, True) is KindBackend.RULES
    assert resolve_kind_backend(_cfg(backend="nonsense"), Mode.A, None, False) \
        is KindBackend.RULES


def test_llm_backend_uses_the_extractor_hint_else_rules() -> None:
    hinted = AtomicFact(content="plain fact text", memory_kind=K.PROCEDURE.value,
                        memory_kind_source=KindSource.MODEL_LLM.value,
                        memory_kind_recipe="kinds-v1:llm")
    unhinted = AtomicFact(content="Never run rm -rf with a glob.")
    out = _clf(mode=Mode.B, llm=True).suggest([hinted, unhinted], caller_kind=None)
    assert out[0].kind is K.PROCEDURE and out[0].source is KindSource.MODEL_LLM
    assert out[1].kind is K.RULE and out[1].source is KindSource.RULES


# -- the model path -----------------------------------------------------------

def test_laya_answer_becomes_a_model_suggestion() -> None:
    laya = _FakeJudge("laya", answers=[_answer(K.STATUS.value, 0.33)])
    out = _clf(judge=laya).suggest(_facts("plain fact text"), caller_kind=None)
    assert out[0] == KindAssignment(K.STATUS, KindSource.MODEL_LAYA, 0.33, KINDS_V1.recipe_id)
    assert laya.calls[0]["recipe"] is KINDS_V1


def test_a_strong_cue_beats_a_disagreeing_model_at_save_as_in_a_run() -> None:
    # Same decision as a classification run (memory_kind_backfill_plan.decide):
    # on short text where a cue fired, the rules' kind is kept against a model
    # that disagrees, so a memory gets the same kind however it was typed.
    laya = _FakeJudge("laya", answers=lambda docs: [_answer(K.SEMANTIC.value, 0.9)] * len(docs))
    out = _clf(judge=laya).suggest(_facts("We decided to ship on Friday.", "plain fact text"),
                                   caller_kind=None)
    assert out[0].kind is K.DECISION and out[0].source is KindSource.RULES
    assert out[1] == KindAssignment(K.SEMANTIC, KindSource.MODEL_LAYA, 0.9, KINDS_V1.recipe_id)
    agreeing = _FakeJudge("laya", answers=[_answer(K.DECISION.value, 0.8)])
    out = _clf(judge=agreeing).suggest(_facts("We decided to ship on Friday."),
                                       caller_kind=None)
    assert out[0].source is KindSource.MODEL_LAYA, "an agreeing model keeps its confidence"


def test_a_strong_cue_beats_a_disagreeing_extractor_hint() -> None:
    hinted = AtomicFact(content="Never run rm -rf with a glob.",
                        memory_kind=K.PROCEDURE.value,
                        memory_kind_source=KindSource.MODEL_LLM.value,
                        memory_kind_recipe="kinds-v1:llm")
    out = _clf(mode=Mode.B, llm=True).suggest([hinted], caller_kind=None)
    assert out[0].kind is K.RULE and out[0].source is KindSource.RULES


@pytest.mark.parametrize("failure", [
    None, [], RuntimeError("worker died"), TimeoutError(), "not-a-list",
    lambda docs: [_answer(K.OPINION.value)] * (len(docs) + 1),
    lambda docs: [object()] * len(docs),
    lambda docs: [_answer("no-such-kind")] * len(docs),
])
def test_model_failure_falls_back_to_rules(failure: object) -> None:
    laya = _FakeJudge("laya", answers=failure)
    out = _clf(judge=laya).suggest(_facts("Never run rm -rf with a glob.", "a plain fact"),
                                   caller_kind=None)
    assert [a.source for a in out] == [KindSource.RULES, KindSource.RULES]
    assert out[0].kind is K.RULE and out[0].recipe == RULES_RECIPE


def test_suggest_never_raises() -> None:
    def boom():
        raise RuntimeError("judge supplier exploded")

    clf = KindClassifier(config=_cfg(), mode=Mode.A, judge_supplier=boom, llm_available=False)
    out = clf.suggest(_facts("Never run rm -rf with a glob."), caller_kind=None)
    assert out[0].kind is K.RULE
    weird = [SimpleNamespace(content=None, fact_type=None), SimpleNamespace()]
    out = _clf(judge=_FakeJudge("laya", answers=RuntimeError())).suggest(
        weird, caller_kind=None)  # type: ignore[arg-type]
    assert len(out) == 2 and all(isinstance(a.kind, MemoryKind) for a in out)
    assert _clf(_cfg(backend=None, enabled="yes")).suggest([], caller_kind=None) == []


def test_correction_needs_cue_and_verify_at_0_8() -> None:
    texts = ("Correction: the API limit is 100 per minute.",   # cue fires
             "The API limit is 100 per minute.")                # no cue
    def run(verify: float | None, choice: str = K.SEMANTIC.value):
        laya = _FakeJudge("laya", answers=lambda docs: [
            _answer(choice, 0.5, verify), _answer(choice, 0.5, None)])
        out = _clf(judge=laya).suggest(_facts(*texts), caller_kind=None)
        return out, laya.calls[0]

    out, call = run(0.8)
    assert call["verify"] == [0], "only cue-fired facts are verified"
    assert out[0].kind is K.CORRECTION and out[0].source is KindSource.MODEL_LAYA
    assert out[1].kind is K.SEMANTIC
    out, _ = run(0.79)
    assert out[0].kind is K.SEMANTIC, "verify below 0.8 keeps the model's own choice"
    out, _ = run(None)
    assert out[0].kind is K.SEMANTIC
    # The model itself can never say correction: the label is not offered.
    assert K.CORRECTION.value not in KINDS_V1.criteria


def test_no_judge_is_ever_built(monkeypatch) -> None:
    from superlocalmemory.core import judge_selection
    from superlocalmemory.retrieval import jev_judge, sufficiency

    def refuse(*_a, **_k):
        raise AssertionError("typing must never build a judge")

    monkeypatch.setattr(judge_selection, "build_sufficiency_judge", refuse)
    monkeypatch.setattr(judge_selection, "swap_sufficiency_judge", refuse)
    monkeypatch.setattr(sufficiency.LayaSufficiencyJudge, "__init__", refuse)
    monkeypatch.setattr(jev_judge.JevSufficiencyJudge, "__init__", refuse)
    for backend in ("auto", "laya", "jev", "llm", "rules"):
        out = _clf(_cfg(backend=backend, jev_consent=True)).suggest(
            _facts("We chose Postgres over DynamoDB."), caller_kind=None)
        assert out[0].kind is K.DECISION


def test_only_first_8_facts_go_to_laya() -> None:
    laya = _FakeJudge("laya", answers=lambda docs: [_answer(K.OPINION.value)] * len(docs))
    facts = _facts(*[f"plain fact number {i}" for i in range(12)])
    out = _clf(judge=laya).suggest(facts, caller_kind=None)
    assert MAX_MODEL_FACTS == 8
    assert len(laya.calls) == 1 and len(laya.calls[0]["documents"]) == 8
    assert [a.source for a in out[:8]] == [KindSource.MODEL_LAYA] * 8
    assert [a.source for a in out[8:]] == [KindSource.RULES] * 4


# -- applying a result to a fact ----------------------------------------------

def test_with_kind_sets_only_the_kind_fields_and_none_clears_a_hint() -> None:
    fact = AtomicFact(content="x" * 10, fact_type=FactType.EPISODIC, scope="global",
                      shared_with=["p2"], pinned=True, memory_kind=K.RULE.value,
                      memory_kind_source=KindSource.MODEL_LLM.value)
    typed = with_kind(fact, KindAssignment(K.DECISION, KindSource.RULES, None, RULES_RECIPE),
                      "2026-10-03T00:00:00+00:00")
    assert (typed.memory_kind, typed.memory_kind_source, typed.memory_kind_recipe,
            typed.memory_kind_at) == (K.DECISION.value, KindSource.RULES.value, RULES_RECIPE,
                                      "2026-10-03T00:00:00+00:00")
    assert typed.fact_type is FactType.EPISODIC, "a suggestion never changes fact_type"
    assert (typed.scope, typed.shared_with, typed.pinned, typed.fact_id) == (
        "global", ["p2"], True, fact.fact_id)
    cleared = with_kind(fact, None, "now")
    assert cleared.memory_kind is None and cleared.memory_kind_source is None
    assert fact.memory_kind == K.RULE.value, "the input fact is never mutated"
