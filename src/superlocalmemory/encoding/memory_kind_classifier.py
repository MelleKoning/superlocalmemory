# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which kind each new fact gets, and from whom (LLD §3.1, §3.3, §3.7).

Source priority, highest first:

1. the kind the caller declared on the write — confirmed, applies to every fact;
2. (a user's later edit — handled by the mutation path, not here);
3. one model suggestion, from the model that is **already running**:
   the Mode B/C extraction call (no extra call), or the answer check's live
   Laya worker, or — only with its own literal-True consent and only when the
   answer check is already Jev — Jev;
4. the cue rules, which always run and cost nothing.

Three rules this module exists to keep:

* **Never start anything (I5).** It reads the live answer-check judge through
  ``judge_supplier`` and never builds, swaps or warms one. No judge, or a judge
  that is not ready, means rules.
* **Never fail a write (I1).** ``suggest`` never raises; any trouble with a
  model falls back to the rules suggestion for every fact.
* **Never decide behaviour.** Everything but the caller's kind is a
  suggestion (source ``model:*`` or ``rules``); measured on real memories the
  best model reached 43 %, so the store shows it as suggested and nothing acts
  on it until a person or a caller confirms it.

Called from the background materializer only. Even if it were called
on a recall thread, no model would run: both model clients refuse to ask
outside ``recall_gate.background_work()``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, Protocol

from superlocalmemory.core import judge_selection
from superlocalmemory.encoding.memory_kind_recipe import (
    KINDS_V1,
    LLM_RECIPE,
    KindAnswer,
    KindRecipe,
)
from superlocalmemory.encoding.memory_kind_rules import cue_kind, suggest_by_rules
from superlocalmemory.storage.memory_kinds import (
    KIND_COLUMNS,
    KindAssignment,
    KindSource,
    MemoryKind,
    parse_kind,
)
from superlocalmemory.storage.models import AtomicFact, Mode

logger = logging.getLogger(__name__)

#: Facts per memory per pass that may go to a model (LLD §3.2): Laya costs about
#: 55 ms a fact, so eight keep one memory's typing well under half a second.
MAX_MODEL_FACTS = 8

#: The caller's kind needs no recipe; this marks where it came from.
CALLER_RECIPE = "caller"

#: Model sources whose answers are weighed against the cue rules (``decide``).
MODEL_SOURCES = frozenset({KindSource.MODEL_LAYA, KindSource.MODEL_JEV, KindSource.MODEL_LLM})


def decide(model: KindAssignment | None, rules: KindAssignment,
           cue: MemoryKind | None) -> KindAssignment | None:
    """The suggestion kept for one fact when a model answered.

    The same rule when a memory is saved and in a classification run, so a
    memory gets one kind however it was typed. A **strong cue** is a rules cue
    that fired on short text (``memory_kind_rules.cue_kind``). On such text the
    cue rules scored 0.887 on the persona set against 0.48 for the real Laya
    model, and no model confidence is calibrated yet, so a model answer that
    disagrees with a strong cue does not replace it. Where no cue fired, the
    model fills the gap. An agreeing model is kept for its confidence.
    ``correction`` from Laya or Jev already went through its own cue-and-verify
    check in ``_merge`` before reaching here, so that answer stands; when the
    verify check failed, the model's other choice does NOT replace the
    correction cue (4.1.22: "a low-quality model suggestion never
    replaces a strong rules cue") - the rules suggestion stands. The Mode
    B/C extraction call never ran that check (L2-11: no extra model call, no
    separate yes/no question) - its own say-so is never enough, so its
    ``correction`` is refused here too and the rules suggestion is kept
    instead, exactly as if the model had failed outright.
    """
    if model is None:
        return None
    if model.kind is MemoryKind.CORRECTION and model.source is KindSource.MODEL_LLM:
        return rules
    if model.source not in MODEL_SOURCES or cue is None:
        return model
    if model.kind is cue:
        return model
    return rules


def _decided(model: KindAssignment | None, rules: KindAssignment,
             fact: Any) -> KindAssignment | None:
    return decide(model, rules, cue_kind(getattr(fact, "content", None)))


class KindBackend(str, Enum):
    OFF = "off"
    RULES = "rules"
    LAYA = "laya"
    JEV = "jev"
    LLM = "llm"


class KindConfigLike(Protocol):
    """The fields of ``MemoryKindConfig`` this module reads."""

    enabled: bool
    backend: str
    jev_consent: bool


_MODEL_SOURCE = {KindBackend.LAYA: KindSource.MODEL_LAYA, KindBackend.JEV: KindSource.MODEL_JEV}


def _setting(cfg: Any, name: str, default: Any) -> Any:
    try:
        return getattr(cfg, name, default)
    except Exception:  # noqa: BLE001 — a broken config object reads as the default
        return default


def resolve_kind_backend(cfg: KindConfigLike, mode: Mode, judge: Any | None,
                         llm_available: bool) -> KindBackend:
    """The one backend typing may use now. Never one that is not already running.

    ``auto`` never picks Jev: typing sends every memory off the device, so it
    happens only when someone chose Jev for typing *and* gave that separate
    consent as the literal boolean ``True``.
    """
    if _setting(cfg, "enabled", True) is not True:
        return KindBackend.OFF
    backend = _setting(cfg, "backend", "auto")
    backend = backend.strip().lower() if isinstance(backend, str) else "auto"
    if backend == KindBackend.OFF.value:
        return KindBackend.OFF
    judge_backend = judge_selection.backend_of(judge)
    llm_ok = mode in (Mode.B, Mode.C) and llm_available is True
    if backend == KindBackend.RULES.value:
        return KindBackend.RULES
    if backend == KindBackend.LAYA.value:
        return KindBackend.LAYA if judge_backend == KindBackend.LAYA.value else KindBackend.RULES
    if backend == KindBackend.JEV.value:
        consent = _setting(cfg, "jev_consent", False) is True
        return KindBackend.JEV if (judge_backend == KindBackend.JEV.value and consent) \
            else KindBackend.RULES
    if backend == KindBackend.LLM.value:
        return KindBackend.LLM if llm_ok else KindBackend.RULES
    if backend != "auto":
        logger.info("memory kinds: unknown backend setting; using the rules")
    if llm_ok:
        return KindBackend.LLM
    if judge_backend == KindBackend.LAYA.value:
        return KindBackend.LAYA
    return KindBackend.RULES


def llm_hint(fact: Any) -> KindAssignment | None:
    """The kind the Mode B/C extraction call attached to ``fact``, if any.

    ``correction`` (offered to the model in ``KIND_INSTRUCTION``, encoding/
    llm_kind_hint.py) is refused here regardless of what the model said: this
    single extraction call carries no separate yes/no check and no verify
    score, so it can never clear the cue-and-verify gate every other backend
    enforces before granting that kind (``decide`` below is a second,
    defence-in-depth refusal for the same reason). ``None`` here falls
    through to the rules suggestion (``_suggest``'s ``or r``).
    """
    if getattr(fact, "memory_kind_source", None) != KindSource.MODEL_LLM.value:
        return None
    kind = parse_kind(getattr(fact, "memory_kind", None))
    if kind is None or kind is MemoryKind.CORRECTION:
        return None
    return KindAssignment(kind, KindSource.MODEL_LLM, None, LLM_RECIPE)


def _rules_for(fact: Any) -> KindAssignment:
    content = getattr(fact, "content", "")
    fact_type = getattr(fact, "fact_type", None)
    return suggest_by_rules(content if isinstance(content, str) else "",
                            getattr(fact_type, "value", fact_type))  # type: ignore[arg-type]


def _merge(answer: KindAnswer, rules: KindAssignment, recipe: KindRecipe,
           source: KindSource) -> KindAssignment:
    """A model answer as a suggestion. ``correction`` only on cue AND verify >= threshold."""
    kind = parse_kind(answer.choice)
    if kind is None or answer.choice not in recipe.criteria:
        raise ValueError("answer outside the recipe")
    if (rules.kind is MemoryKind.CORRECTION and answer.verify is not None
            and answer.verify >= recipe.correction_threshold):
        kind = MemoryKind.CORRECTION
    return KindAssignment(kind, source, float(answer.confidence), recipe.recipe_id)


class KindClassifier:
    """Suggests a kind per fact. One instance per engine; holds no model of its own."""

    def __init__(self, *, config: KindConfigLike, mode: Mode,
                 judge_supplier: Callable[[], Any | None], llm_available: bool,
                 recipe: KindRecipe = KINDS_V1) -> None:
        self._config = config
        self._mode = mode
        self._judge_supplier = judge_supplier
        self._llm_available = llm_available
        self._recipe = recipe

    def suggest(self, facts: Sequence[AtomicFact], *,
                caller_kind: MemoryKind | None) -> list[KindAssignment | None]:
        """One assignment (or None = leave untyped) per fact, in order. Never raises."""
        facts = list(facts or [])
        if isinstance(caller_kind, MemoryKind):
            return [KindAssignment(caller_kind, KindSource.CALLER, None, CALLER_RECIPE)] \
                * len(facts)
        try:
            return self._suggest(facts)
        except Exception as exc:  # noqa: BLE001 — typing never fails a write (I1)
            logger.warning("memory kinds: suggestion failed (%s); using the rules",
                           type(exc).__name__)
            return [_rules_for(f) for f in facts]

    def _suggest(self, facts: list[Any]) -> list[KindAssignment | None]:
        judge = self._live_judge()
        backend = resolve_kind_backend(self._config, self._mode, judge, self._llm_available)
        if backend is KindBackend.OFF:
            return [None] * len(facts)
        rules = [_rules_for(f) for f in facts]
        out: list[KindAssignment | None] = list(rules)
        if backend is KindBackend.LLM:
            return [_decided(llm_hint(f), r, f) or r for f, r in zip(facts, rules)]
        if backend in _MODEL_SOURCE and facts:
            asked = self._ask(backend, judge, facts[:MAX_MODEL_FACTS], rules)
            if asked:
                out[:len(asked)] = [_decided(a, r, f)
                                    for a, r, f in zip(asked, rules, facts)]
        return out

    def _live_judge(self) -> Any | None:
        try:
            return self._judge_supplier()
        except Exception as exc:  # noqa: BLE001 — no judge is a valid answer
            logger.info("memory kinds: the answer-check judge could not be read (%s)",
                        type(exc).__name__)
            return None

    def _ask(self, backend: KindBackend, judge: Any, facts: list[Any],
             rules: list[KindAssignment]) -> list[KindAssignment] | None:
        """Model suggestions for ``facts``, or None on any failure (then: rules)."""
        if judge is None or getattr(judge, "ready", False) is not True:
            return None
        documents = [getattr(f, "content", "") for f in facts]
        if not all(isinstance(d, str) and d.strip() for d in documents):
            return None
        verify = [i for i, r in enumerate(rules[:len(facts)])
                  if r.kind is MemoryKind.CORRECTION]
        try:
            if backend is KindBackend.JEV:
                answers = judge.ask_kinds(documents, self._recipe, verify,
                                          consent=_setting(self._config, "jev_consent",
                                                           False) is True)
            else:
                answers = judge.ask_kinds(documents, self._recipe, verify)
            if not isinstance(answers, list) or len(answers) != len(facts) \
                    or not all(isinstance(a, KindAnswer) for a in answers):
                return None
            source = _MODEL_SOURCE[backend]
            return [_merge(a, r, self._recipe, source) for a, r in zip(answers, rules)]
        except Exception as exc:  # noqa: BLE001 — a model failure costs a suggestion only
            logger.info("memory kinds: %s suggestion unavailable (%s); using the rules",
                        backend.value, type(exc).__name__)
            return None


def with_kind(fact: AtomicFact, assignment: KindAssignment | None,
              now_iso: str) -> AtomicFact:
    """A copy of ``fact`` carrying ``assignment`` in its kind fields (None clears them).

    Touches only the five kind fields — never ``fact_type``, sharing, scope or
    identity. Clearing matters: the Mode B/C extractor attaches a hint, and a
    fact whose backend turned out to be off must not carry it into storage.
    """
    if assignment is None:
        return dataclasses.replace(fact, **{name: None for name in KIND_COLUMNS})
    return dataclasses.replace(fact, **assignment.as_columns(now_iso))


__all__ = [
    "CALLER_RECIPE",
    "MAX_MODEL_FACTS",
    "MODEL_SOURCES",
    "decide",
    "KindAnswer",
    "KindBackend",
    "KindClassifier",
    "KindConfigLike",
    "llm_hint",
    "resolve_kind_backend",
    "with_kind",
]
