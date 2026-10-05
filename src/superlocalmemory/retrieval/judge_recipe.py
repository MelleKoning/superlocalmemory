# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What the answer check asks, and how a memory is shown to it.

Laya and Jev are typed decision models, not chat models: they answer one
precisely worded question about one piece of text. The wording and the way a
memory is rendered decide how good the judgement is, so both live here, under
one versioned id, and both backends read the same recipe. A verdict names the
recipe it came from, so a measurement taken under one recipe is never read as
evidence for another.

What is sent, exactly: the question, the recipe's wording, and for each memory
its text, cut to ``max_document_chars`` (the hosted check redacts credentials
first). Nothing else — no type, no date, no entity names. Showing the model
more would be a new recipe, and would need measuring before it could decide
anything.

A threshold belongs to a (backend, recipe) pair and to nothing else. Laya's
was measured on Laya; Jev's has not been measured by SLM at all, and the
response says so rather than borrowing a number.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeDocument:
    """One recalled memory, as the answer check sees it: its text, and only that."""

    content: str


@dataclass(frozen=True)
class JudgeRecipe:
    """A versioned way of asking "does this memory answer the question"."""

    recipe_id: str
    question: str
    max_document_chars: int = 1800

    def render(self, document: JudgeDocument) -> str:
        """The text the model reads for one memory: its content, cut to the limit."""
        return document.content[: self.max_document_chars]


@dataclass(frozen=True)
class Calibration:
    """The decision threshold for one backend under one recipe, and its standing."""

    threshold: float
    status: str


RECIPE_V1 = JudgeRecipe(
    recipe_id="sufficiency-v1",
    question=(
        "Read `question` and `memory`. Is this statement true: the memory "
        "contains the specific information that answers the question."
    ),
)

ACTIVE_RECIPE = RECIPE_V1

#: Keyed by (backend, recipe_id). A missing pair means "never measured": the
#: backend may still report a confidence, but cannot abstain on its own say-so.
CALIBRATIONS: dict[tuple[str, str], Calibration] = {
    # 124 questions on a real store; chosen on the same set it was measured on.
    ("laya", "sufficiency-v1"): Calibration(0.6, "measured_small_sample_not_calibrated"),
    # Jev's own published decision rule, then measured by SLM on a public,
    # synthetic, persona-mixed set (178 held-out questions): it keeps 95% of
    # sets that hold the answer and flags 91% of those that do not. Real
    # memories are never sent for measurement, so this is the synthetic figure.
    ("jev", "sufficiency-v1"): Calibration(0.5, "measured_synthetic_set_not_calibrated"),
    # The same check on a REORDERED top three (one request that also chooses
    # the order of up to 30 memories). Its own entry: the figure above was
    # measured on three memories and does not carry over. Measured by SLM on a
    # public long-conversation benchmark, never on real memories.
    ("jev-listwise", "sufficiency-v1"): Calibration(
        0.5, "measured_public_benchmark_not_calibrated"),
}


#: For a pair nobody measured. Every valid probability is >= 0, so
#: ``confidence < 0.0`` is never true: the verdict still reports how confident
#: the model was, but it cannot make a recall abstain on a number nobody tested.
UNMEASURED = Calibration(0.0, "not_measured_cannot_abstain")


def calibration_for(backend: str, recipe: JudgeRecipe = ACTIVE_RECIPE) -> Calibration | None:
    """This store's measured threshold when there is one, else the built-in.

    A measured threshold comes from ``answer_check_calibration.json`` in the
    data folder (see ``judge_calibration_file``) and replaces the built-in
    figure for that one (backend, recipe) pair only.
    """
    from superlocalmemory.retrieval.judge_calibration_file import (
        STATUS,
        measured_thresholds,
    )

    measured = measured_thresholds().get((backend, recipe.recipe_id))
    if measured is not None:
        return Calibration(measured, STATUS)
    return CALIBRATIONS.get((backend, recipe.recipe_id))


def calibration_or_unmeasured(backend: str,
                              recipe: JudgeRecipe = ACTIVE_RECIPE) -> Calibration:
    """The measured calibration, or one that can never abstain."""
    return calibration_for(backend, recipe) or UNMEASURED


def coerce_documents(items: Iterable[object]) -> tuple[JudgeDocument, ...]:
    """Documents as given. A plain string, the shape callers used before the
    recipe existed, is wrapped; anything else is dropped rather than judged."""
    out: list[JudgeDocument] = []
    for item in items:
        if isinstance(item, JudgeDocument):
            out.append(item)
        elif isinstance(item, str):
            out.append(JudgeDocument(content=item))
    return tuple(out)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def document_from_fact(fact: object) -> JudgeDocument:
    """One recalled fact as the answer check sees it: its text. Never raises."""
    return JudgeDocument(content=_text(getattr(fact, "content", "")))


__all__ = [
    "ACTIVE_RECIPE",
    "CALIBRATIONS",
    "Calibration",
    "JudgeDocument",
    "JudgeRecipe",
    "RECIPE_V1",
    "UNMEASURED",
    "calibration_for",
    "calibration_or_unmeasured",
    "coerce_documents",
    "document_from_fact",
]
