# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The kind question asked of a model, and what is known about each model's answers.

A recipe is data, not code: the exact wording, the labels offered, and the
separate yes/no check that alone may promote a suggestion to ``correction``.
Changing a word changes the measured numbers, so a new wording is a new
``recipe_id`` with its own calibration row — never an edit to ``KINDS_V1``.

Measured on 83 hand-labelled real memories (4.1.19 research §1): the best
model design reached 43 % accuracy. That is why every answer from a recipe is
a *suggestion*: stored with its source and confidence, shown as "suggested",
and never allowed to change behaviour (LLD I6: ``behaviour_min_confidence`` is
``None`` for every row below).

Pure data. No worker, judge or network import, so retrieval clients and the
encoding classifier can both import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from superlocalmemory.storage.memory_kinds import MemoryKind

K = MemoryKind

#: Status strings, shared with the answer check's calibration vocabulary.
STATUS_NOT_MEASURED = "not_measured"
STATUS_SMALL_SAMPLE = "measured_small_sample_not_calibrated"

#: The display threshold the E1 run selected (SEL_F9_conf0.2: 0.434 accuracy).
DEFAULT_DISPLAY_MIN_CONFIDENCE = 0.20

#: Recipe id stamped on a kind the Mode B/C extraction call suggested.
LLM_RECIPE = "kinds-v1:llm"


@dataclass(frozen=True, slots=True)
class KindRecipe:
    """One wording of the "what kind of memory is this?" question."""

    recipe_id: str
    instructions: str
    #: label -> plain-language description. Read-only; label order is the
    #: order offered to the model.
    criteria: Mapping[str, str]
    #: Yes/no statement asked only when the rules saw a strong correction cue.
    correction_verify: str
    correction_threshold: float
    #: Each memory is cut to this many characters before it is asked about.
    max_chars: int

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.criteria)


@dataclass(frozen=True, slots=True)
class KindAnswer:
    """A model's answer about one memory, already validated against its recipe."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    verify: float | None


@dataclass(frozen=True, slots=True)
class KindCalibration:
    """What is known about one (backend, recipe) pair's answers."""

    backend: str
    recipe_id: str
    status: str
    display_min_confidence: float
    #: The confidence at which a suggestion may drive behaviour. ``None`` =
    #: never (LLD I6). Every row ships ``None`` in 4.1.19.
    behaviour_min_confidence: float | None


KINDS_V1 = KindRecipe(
    recipe_id="kinds-v1",
    instructions="What kind of memory is this? Pick the one that fits best.",
    # Eight labels. "correction" is deliberately absent: offered as a label,
    # Laya used it as a sink (8 of 11 plain facts came back as corrections).
    criteria=MappingProxyType({
        K.SEMANTIC.value: "a lasting fact about how something is or works",
        K.EPISODIC.value: "something that happened at a particular time",
        K.STATUS.value: "the current state of ongoing work; will go out of date",
        K.OPINION.value: "someone's preference, taste, or judgement",
        K.RULE.value: "a standing instruction to always or never do something",
        K.DECISION.value: "a choice that was made, often with its reason",
        K.PROCEDURE.value: "steps or commands for how to do something",
        K.PROSPECTIVE.value: "something planned or still to be done",
    }),
    correction_verify=("This memory says an earlier statement was wrong and gives the "
                       "corrected version."),
    correction_threshold=0.8,
    max_chars=1800,
)

#: Every recipe this build knows, by id. A later recipe (for example one asking
#: whether a new memory makes an old one out of date) is added here with its
#: own id and its own calibration rows.
RECIPES: Mapping[str, KindRecipe] = MappingProxyType({KINDS_V1.recipe_id: KINDS_V1})


def _row(backend: str, recipe_id: str, status: str) -> KindCalibration:
    return KindCalibration(backend, recipe_id, status, DEFAULT_DISPLAY_MIN_CONFIDENCE, None)


#: Recipe id stamped on a cue-rules suggestion (``memory_kind_rules``).
RULES_RECIPE = "kinds-rules-v1"

KIND_CALIBRATIONS: Mapping[tuple[str, str], KindCalibration] = MappingProxyType({
    ("laya", KINDS_V1.recipe_id): _row("laya", KINDS_V1.recipe_id, STATUS_SMALL_SAMPLE),
    ("jev", KINDS_V1.recipe_id): _row("jev", KINDS_V1.recipe_id, STATUS_NOT_MEASURED),
    ("llm", LLM_RECIPE): _row("llm", LLM_RECIPE, STATUS_NOT_MEASURED),
    ("rules", RULES_RECIPE): _row("rules", RULES_RECIPE, STATUS_SMALL_SAMPLE),
})


def calibration_for(backend: str, recipe_id: str) -> KindCalibration:
    """The calibration row for a pair, or a not-measured one. Never raises."""
    row = KIND_CALIBRATIONS.get((backend, recipe_id)) if isinstance(backend, str) \
        and isinstance(recipe_id, str) else None
    if row is not None:
        return row
    return _row(str(backend), str(recipe_id), STATUS_NOT_MEASURED)


__all__ = [
    "DEFAULT_DISPLAY_MIN_CONFIDENCE",
    "KINDS_V1",
    "KIND_CALIBRATIONS",
    "LLM_RECIPE",
    "RECIPES",
    "RULES_RECIPE",
    "STATUS_NOT_MEASURED",
    "STATUS_SMALL_SAMPLE",
    "KindAnswer",
    "KindCalibration",
    "KindRecipe",
    "calibration_for",
]
