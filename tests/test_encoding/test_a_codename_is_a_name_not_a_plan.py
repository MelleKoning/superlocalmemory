# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A memory that names something ("X is the codename for Y") is a fact, not a plan.

Seen in 4.1.21 and reproduced on the pinned Laya weights in 4.1.22: "Heron is the
codename for the new mobile app." was suggested as ``prospective`` (Laya
confidence 0.77), because no cue rule fired and the model filled the gap. A
naming cue is now a strong rules cue for ``semantic``, so a disagreeing model
suggestion no longer replaces it (``memory_kind_classifier.decide``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.encoding.memory_kind_classifier import KindClassifier
from superlocalmemory.encoding.memory_kind_recipe import KindAnswer
from superlocalmemory.encoding.memory_kind_rules import cue_kind
from superlocalmemory.storage.memory_kinds import KindSource, MemoryKind
from superlocalmemory.storage.models import AtomicFact, Mode

NAMING = [
    "Heron is the codename for the new mobile app.",
    "Bramble Point is our codename for the warehouse migration plan.",
    "Project Lark is the internal code name of the billing rewrite.",
    "QX-7 stands for the quarterly export job.",
    "'Atlas' is short for the Atlas logistics platform.",
]


@pytest.mark.parametrize("text", NAMING)
def test_a_naming_sentence_is_a_semantic_cue(text) -> None:
    assert cue_kind(text) is MemoryKind.SEMANTIC


@pytest.mark.parametrize("text,kind", [
    ("We plan to pick a codename for the app next week.", MemoryKind.PROSPECTIVE),
    ("Never use the codename in customer emails.", MemoryKind.RULE),
    ("Correction: the codename is Heron, not Egret.", MemoryKind.CORRECTION),
    ("We decided Heron is the codename for the app.", MemoryKind.DECISION),
])
def test_stronger_cues_still_win(text, kind) -> None:
    assert cue_kind(text) is kind


class _Laya:
    backend = "laya"
    ready = True

    def ask_kinds(self, documents, recipe, verify_indices):
        return [KindAnswer(choice="prospective", probabilities={"prospective": 0.77},
                           confidence=0.77, verify=None) for _ in documents]


def test_a_model_plan_suggestion_no_longer_replaces_the_naming_cue(monkeypatch) -> None:
    from superlocalmemory.core import judge_selection

    monkeypatch.setattr(judge_selection, "backend_of", lambda judge: "laya")
    clf = KindClassifier(config=SimpleNamespace(enabled=True, backend="laya",
                                                jev_consent=False),
                         mode=Mode.A, judge_supplier=_Laya, llm_available=False)
    out = clf.suggest([AtomicFact(content=NAMING[0])], caller_kind=None)
    assert out[0].kind is MemoryKind.SEMANTIC
    assert out[0].source is KindSource.RULES
