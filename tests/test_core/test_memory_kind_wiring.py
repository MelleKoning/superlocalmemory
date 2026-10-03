# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The engine owns one kind classifier, built from what is already running.

It reads the live answer-check judge (a judge switch takes effect at once,
without rebuilding anything) and the live settings (turning memory kinds off
in the dashboard stops suggestions on the next memory), and it never builds a
judge, a worker or a client of its own (LLD I5).
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.core import judge_selection
from superlocalmemory.core.memory_kind_config import MemoryKindConfig, memory_kind_config_from
from superlocalmemory.core.memory_kind_wiring import build_kind_classifier
from superlocalmemory.encoding.memory_kind_classifier import KindClassifier
from superlocalmemory.encoding.memory_kind_recipe import KindAnswer
from superlocalmemory.storage.memory_kinds import KindSource, MemoryKind
from superlocalmemory.storage.models import AtomicFact, Mode


class _Judge:
    backend = "laya"
    ready = True

    def __init__(self) -> None:
        self.calls = 0

    def ask_kinds(self, documents, recipe, verify):
        self.calls += 1
        return [KindAnswer("status", {"status": 0.7}, 0.7, None) for _ in documents]


def _engine(judge=None, cfg: MemoryKindConfig | None = None, mode=Mode.A, llm=None):
    return SimpleNamespace(
        _config=SimpleNamespace(mode=mode, memory_kinds=cfg or MemoryKindConfig()),
        _retrieval_engine=SimpleNamespace(_sufficiency_judge=judge),
        _llm=llm,
    )


def _facts(*texts: str) -> list[AtomicFact]:
    return [AtomicFact(profile_id="default", memory_id="m", content=t) for t in texts]


def test_engine_builds_a_kind_classifier(engine_with_mock_deps) -> None:
    assert isinstance(engine_with_mock_deps._kind_classifier, KindClassifier)
    out = engine_with_mock_deps._kind_classifier.suggest(
        _facts("Never push to main without a review."), caller_kind=None)
    assert out[0].kind is MemoryKind.RULE and out[0].source is KindSource.RULES


def test_reads_the_live_judge_and_the_live_settings() -> None:
    engine = _engine()
    classifier = build_kind_classifier(engine)
    assert classifier.suggest(_facts("The office is in Pune."), caller_kind=None)[0].source \
        is KindSource.RULES
    judge = _Judge()
    engine._retrieval_engine._sufficiency_judge = judge   # the answer check switched to Laya
    from superlocalmemory.core import recall_gate
    with recall_gate.background_work():
        out = classifier.suggest(_facts("The office is in Pune."), caller_kind=None)
    assert judge.calls == 1 and out[0].source is KindSource.MODEL_LAYA
    engine._config.memory_kinds = memory_kind_config_from({"enabled": False})
    assert classifier.suggest(_facts("x"), caller_kind=None) == [None]


def test_a_replaced_retrieval_engine_is_followed() -> None:
    engine = _engine()
    classifier = build_kind_classifier(engine)
    judge = _Judge()
    engine._retrieval_engine = SimpleNamespace(_sufficiency_judge=judge)
    from superlocalmemory.core import recall_gate
    with recall_gate.background_work():
        classifier.suggest(_facts("The office is in Pune."), caller_kind=None)
    assert judge.calls == 1


def test_no_judge_is_ever_built(monkeypatch) -> None:
    def refuse(*a, **k):
        raise AssertionError("typing must never build a judge")

    monkeypatch.setattr(judge_selection, "build_sufficiency_judge", refuse)
    classifier = build_kind_classifier(_engine())
    assert classifier.suggest(_facts("Never push."), caller_kind=None)[0] is not None


def test_mode_b_with_a_model_uses_the_extraction_hint() -> None:
    engine = _engine(mode=Mode.B, llm=object())
    classifier = build_kind_classifier(engine)
    fact = _facts("The office is in Pune.")[0]
    fact.memory_kind, fact.memory_kind_source = "status", "model:llm"
    assert classifier.suggest([fact], caller_kind=None)[0].source is KindSource.MODEL_LLM
