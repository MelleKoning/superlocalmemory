# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Using a memory's kind to answer the question that was asked.

"What did we decide about X" wants a decision, "how do I ..." wants a how-to,
"what is the current state of X" wants the newest current-state memory. This
module reads that intent from the question with fixed patterns - no model, no
store write - and makes two bounded adjustments to the finished ranking. It may
move only what it has evidence for; every other result keeps exactly the order
the pipeline gave it.

1. A memory whose kind matches the intent may move up past a neighbour with
   less kind evidence, compared on the pipeline's own ranking key
   (``core.recall_pipeline._rank_key``), when its key lifted by the boost
   (default 15% of its magnitude, never more than 50%) beats the neighbour's.
   It can never jump a clearly better result. A confirmed kind counts fully, a
   suggested one by half: a model's suggestion is a weaker signal than the kind
   its author declared.
2. When the question asks for a current state, a decision, a rule, or a value
   as it stands now ("what is the X", "which X do we use", "how many X"), a
   newer memory is placed above an older one only when BOTH carry that kind
   CONFIRMED (by a
   user or a caller - LLD I6, Varun's V2) and both mention an entity the
   QUESTION names. The older one then says which memory is newer. A suggested
   kind never drives this, and neither does an entity the question does not
   name. Nothing is removed or hidden.

Neither adjustment moves anything above the exact lexical hit recall pins first
(``retrieval.exact_lexical``). A question that names no intent, or where no
memory's kind matches it, is returned in exactly the order it came in.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, Union

from superlocalmemory.retrieval.exact_lexical import is_exact_lexical_hit, normalize_query
from superlocalmemory.storage.memory_kinds import MemoryKind, is_confirmed

logger = logging.getLogger(__name__)

#: Default ceiling on the kind boost: a matching memory can pass a neighbour
#: whose ranking key is at most this fraction (of its own) higher.
DEFAULT_BOOST = 0.15
#: The most the boost may ever be, whatever the configuration says.
MAX_BOOST = 0.5
#: Only the head of the list is re-ordered; below it nobody reads the order.
WINDOW = 10

_I = re.IGNORECASE
_CUES: tuple[tuple[MemoryKind, re.Pattern[str]], ...] = (
    (MemoryKind.DECISION, re.compile(
        r"\b(decid\w*|decision\w*|chose|chosen|choose|settled|agreed|went with)\b", _I)),
    (MemoryKind.PROCEDURE, re.compile(
        r"(^|\b)how (do|can|should|to|did) (i|we|you)\b|\bhow to\b|\bsteps? (to|for)\b"
        r"|\bcommands? (to|for)\b|\binstructions?\b|\bprocedure\b", _I)),
    (MemoryKind.STATUS, re.compile(
        r"\b(current(ly)?|latest|status|state of|where (are|is) (we|it|things)|progress"
        r"|so far|right now|as of now|at the moment)\b", _I)),
    (MemoryKind.RULE, re.compile(
        r"\b(rules?|polic(y|ies)|conventions?|guidelines?|am i allowed|are we allowed"
        r"|must (i|we)|should (i|we) (ever|always|never))\b", _I)),
    (MemoryKind.PROSPECTIVE, re.compile(
        r"\b(plan(ned|s)?|to-?dos?|next steps?|upcoming|scheduled|deadlines?|going to"
        r"|what('s| is) next)\b", _I)),
    (MemoryKind.OPINION, re.compile(
        r"\b(prefer(ence|s|red)?|favou?rite|opinion|what do (i|you) think)\b", _I)),
    (MemoryKind.CORRECTION, re.compile(
        r"\b(correct(ion|ed)|was wrong|mistaken?|erratum)\b", _I)),
)
_LATEST_KINDS = frozenset({MemoryKind.STATUS, MemoryKind.DECISION, MemoryKind.RULE})

#: A question about a value as it stands now ("what is the recall ceiling",
#: "which editor do we use", "how many workers"). It asks for the newest of
#: two confirmed decisions, rules or statuses about what it names, but carries
#: no kind of its own, so it never triggers the kind boost. "What was ..."
#: asks about the past and is not one.
_CURRENT_VALUE = re.compile(
    r"\b(what|which)(\s+is|\s+are|'s|'re)\s+(the|our|my|your)\b"
    r"|\bwhich\s+\w+(\s+\w+)?\s+(do|does|should)\s+(we|i|you)\s+use\b"
    r"|\bhow\s+(many|much|long|often)\b", _I)

#: The entities a question names: given, or read on demand (and only if needed).
Subject = Union[Iterable[str], Callable[[], Iterable[str]], None]


@dataclass(frozen=True, slots=True)
class Intent:
    kinds: frozenset[MemoryKind]
    #: The question asks for a value as it stands now (see _CURRENT_VALUE).
    current_value: bool = False

    @property
    def wants_latest(self) -> bool:
        return bool(self.kinds & _LATEST_KINDS)

    @property
    def values(self) -> frozenset[str]:
        return frozenset(k.value for k in self.kinds)


def query_intent(query: str) -> Intent:
    """The memory kinds a question asks for (possibly none)."""
    if not isinstance(query, str) or not query.strip():
        return Intent(frozenset())
    return Intent(frozenset(kind for kind, cue in _CUES if cue.search(query)),
                  current_value=bool(_CURRENT_VALUE.search(query)))


def clamp_boost(value: object) -> float:
    """A usable boost: a finite number clamped to [0, MAX_BOOST]; anything else
    (text, a boolean, NaN, infinity, nothing) reads as DEFAULT_BOOST."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_BOOST
    number = float(value)
    if not math.isfinite(number):
        return DEFAULT_BOOST
    return min(MAX_BOOST, max(0.0, number))


def validated_settings(enabled: object, boost: object) -> tuple[bool, float]:
    """``(kind_aware, kind_aware_boost)`` as recall may use them. Only the
    boolean False turns the pass off; any other non-boolean reads as the
    default (on), the rule every other memory-kind switch follows."""
    return (enabled if isinstance(enabled, bool) else True), clamp_boost(boost)


def _kind_weight(fact: Any, intent: Intent) -> float:
    kind = getattr(fact, "memory_kind", None)
    if not kind or kind not in intent.values:
        return 0.0
    return 1.0 if is_confirmed(getattr(fact, "memory_kind_source", None)) else 0.5


def _boosted_order(head: list, intent: Intent, boost: float,
                   floor: int) -> tuple[list[int], set[int]]:
    """Indices of ``head`` in their new order, and which ones the boost moved.

    Each matching memory, in incoming order, moves up one neighbour at a time
    while the neighbour has less kind evidence AND a ranking key below the
    memory's lifted key (key + boost x weight x |key|; relative to |key| so a
    negative learned utility is still lifted, never sunk). Ties keep the
    incoming order. Nothing moves above ``floor``.
    """
    order = list(range(len(head)))
    weights = [_kind_weight(r.fact, intent) for r in head]
    if boost <= 0.0 or not any(weights):
        return order, set()
    from superlocalmemory.core.recall_pipeline import _rank_key

    keys = [-_rank_key(r)[0] for r in head]
    lifted = [k + boost * w * abs(k) for k, w in zip(keys, weights)]
    moved: set[int] = set()
    for item in range(len(head)):
        if not weights[item]:
            continue
        pos = order.index(item)
        while pos > floor:
            above = order[pos - 1]
            if weights[above] >= weights[item] or lifted[above] >= lifted[item]:
                break
            order[pos - 1], order[pos] = item, above
            pos -= 1
            moved.add(item)
    return order, moved


def _subject_ids(subject: Subject) -> frozenset[str]:
    if callable(subject):
        try:
            subject = subject()
        except Exception as exc:  # noqa: BLE001 - ordering is advisory
            logger.debug("kind-aware: the question's subject was unreadable (%s)",
                         type(exc).__name__)
            return frozenset()
    if subject is None or isinstance(subject, str):
        return frozenset()
    return frozenset(str(s) for s in subject)


def _confirmed_latest(fact: Any, wanted: frozenset[str]) -> bool:
    return (getattr(fact, "memory_kind", None) in wanted
            and is_confirmed(getattr(fact, "memory_kind_source", None)))


def _named(fact: Any, subject: frozenset[str]) -> frozenset[str]:
    return subject & frozenset(str(e) for e in (getattr(fact, "canonical_entities", None) or ()))


def _latest_first(head: list, order: list[int], wanted: frozenset[str],
                  subject: frozenset[str], floor: int, notes: dict[int, list[str]]) -> list[int]:
    """A newer memory moves above the older ones it shares a confirmed kind and
    a question-named entity with; each older one it passes is told so.

    One pass in incoming order; a mover never jumps a memory about the same
    subject that is as new or newer, so subjects that do not chain cannot loop.
    """
    eligible = {i for i in order if _confirmed_latest(head[i].fact, wanted)
                and _named(head[i].fact, subject)}
    if len(eligible) < 2:
        return order
    order = list(order)

    def same(a: int, b: int) -> bool:
        fa, fb = head[a].fact, head[b].fact
        return fa.memory_kind == fb.memory_kind and bool(_named(fa, subject) & _named(fb, subject))

    for mover in [i for i in order[floor:] if i in eligible]:
        pos, target = order.index(mover), None
        born = str(head[mover].fact.created_at or "")
        for i in range(pos - 1, floor - 1, -1):
            other = order[i]
            if other not in eligible or not same(other, mover):
                continue
            if str(head[other].fact.created_at or "") >= born:
                break
            target = i
        if target is None:
            continue
        for older in order[target:pos]:
            if older in eligible and same(older, mover):
                notes.setdefault(older, []).append(f"newer:{head[mover].fact.fact_id}")
        order.insert(target, order.pop(pos))
    return order


def _restamped(results: list, head: list, order: list[int],
               notes: dict[int, list[str]]) -> list:
    """New result objects for the new order; the input is not modified."""
    ordered = [*((i, head[i]) for i in order), *((None, r) for r in results[len(head):])]
    out = []
    for rank, (index, r) in enumerate(ordered, start=1):
        chain = list(r.evidence_chain or [])
        chain += [n for n in notes.get(index, ()) if n not in chain]
        out.append(replace(r, rank_position=rank, evidence_chain=chain))
    return out


def apply_kind_awareness(results: list, query: str, *, enabled: bool = True,
                         boost: float = DEFAULT_BOOST, subject: Subject = None) -> list:
    """``results`` re-ordered for the question's intent: the same items, nothing
    added or removed. The input list is returned untouched when disabled, when
    the question names no intent, or when nothing moved.

    ``subject``: the entity ids the question names (or a callable returning
    them, called only when latest-first could apply). Without it latest-first
    never fires: two memories sharing some entity is not evidence that they
    describe the subject that was asked about.
    """
    if not enabled or not results:
        return results
    intent = query_intent(query)
    if not intent.kinds and not intent.current_value:
        return results
    head = list(results[:WINDOW])
    floor = 1 if is_exact_lexical_hit(head[0], normalize_query(query)) else 0
    order, moved = _boosted_order(head, intent, clamp_boost(boost), floor)
    notes: dict[int, list[str]] = {
        i: [f"kind_intent({head[i].fact.memory_kind})"] for i in moved}
    wanted = frozenset(k.value for k in (
        (intent.kinds & _LATEST_KINDS) | (_LATEST_KINDS if intent.current_value else frozenset())))
    if wanted and sum(_confirmed_latest(r.fact, wanted) for r in head[floor:]) >= 2:
        order = _latest_first(head, order, wanted, _subject_ids(subject), floor, notes)
    if order == list(range(len(head))):
        return results
    return _restamped(results, head, order, notes)


def query_subject(query: str, profile_id: str, *, resolver: Any = None,
                  db: Any = None) -> frozenset[str]:
    """The entities the question names, resolved read-only exactly the way the
    ``about`` facet resolves a name (``facets.entity_ids_named``). Empty when it
    names none or the store cannot be read - never an error, never a write."""
    if resolver is None and db is None:
        return frozenset()
    from superlocalmemory.retrieval.entity_channel import extract_query_entities
    from superlocalmemory.retrieval.facets import entity_ids_named

    try:
        names = extract_query_entities(query)
        if not names:
            return frozenset()
        return frozenset(entity_ids_named(db, names, profile_id, resolver))
    except Exception as exc:  # noqa: BLE001 - ordering is advisory
        logger.debug("kind-aware: the question's subject could not be read (%s)",
                     type(exc).__name__)
        return frozenset()


def apply_for_recall(results: list, query: str, profile_id: str, retrieval_config: Any,
                     *, engine: Any = None, db: Any = None) -> list:
    """The pass as recall runs it: settings from ``retrieval.kind_aware`` and
    ``retrieval.kind_aware_boost``, the subject read from the store the
    retrieval used and only when latest-first could apply. Off means the input
    list comes back as it went in."""
    enabled, boost = validated_settings(
        getattr(retrieval_config, "kind_aware", True),
        getattr(retrieval_config, "kind_aware_boost", DEFAULT_BOOST))
    if not enabled:
        return results
    store = getattr(engine, "_db", None)
    store = db if store is None else store
    resolver = getattr(getattr(engine, "_entity", None), "_resolver", None)
    return apply_kind_awareness(
        results, query, boost=boost,
        subject=lambda: query_subject(query, profile_id, resolver=resolver, db=store))


__all__ = [
    "DEFAULT_BOOST", "Intent", "MAX_BOOST", "WINDOW", "apply_for_recall",
    "apply_kind_awareness", "clamp_boost", "query_intent", "query_subject",
    "validated_settings",
]
