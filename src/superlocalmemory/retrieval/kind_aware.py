# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Using a memory's kind to answer the question that was asked.

"What did we decide about X" wants a decision, "how do I ..." wants a how-to,
"what is the current state of X" wants the newest current-state memory. This
module reads that intent from the question with fixed patterns - no model, no
store write - and makes two bounded adjustments to the finished ranking:

1. A memory whose kind matches the intent may move up past a neighbour whose
   score is within the boost (default 15%). It can never jump a clearly better
   result. A confirmed kind counts fully, a suggested one by half: a model's
   suggestion is a weaker signal than the kind its author declared.
2. When the question asks for a current state or a decision, the newer of two
   such memories about the same subject (a shared entity) is placed first, and
   the older one says which memory is newer. Nothing is removed or hidden.

A question that names no intent is returned in exactly the order it came in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from superlocalmemory.storage.memory_kinds import MemoryKind, is_confirmed

#: Default ceiling on the kind boost: a matching memory can pass a neighbour
#: whose score is at most this fraction higher.
DEFAULT_BOOST = 0.15
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
_LATEST_KINDS = frozenset({MemoryKind.STATUS, MemoryKind.DECISION})


@dataclass(frozen=True, slots=True)
class Intent:
    kinds: frozenset[MemoryKind]

    @property
    def wants_latest(self) -> bool:
        return bool(self.kinds & _LATEST_KINDS)


def query_intent(query: str) -> Intent:
    """The memory kinds a question asks for (possibly none)."""
    if not isinstance(query, str) or not query.strip():
        return Intent(frozenset())
    return Intent(frozenset(kind for kind, cue in _CUES if cue.search(query)))


def _kind_weight(fact: Any, intent: Intent) -> float:
    kind = getattr(fact, "memory_kind", None)
    if not kind or kind not in {k.value for k in intent.kinds}:
        return 0.0
    return 1.0 if is_confirmed(getattr(fact, "memory_kind_source", None)) else 0.5


def _boosted_order(results: list, intent: Intent, boost: float) -> list:
    """Stable re-sort of the head by score x (1 + boost x weight)."""
    head, tail = results[:WINDOW], results[WINDOW:]
    keyed = []
    for position, r in enumerate(head):
        weight = _kind_weight(r.fact, intent)
        keyed.append((-(float(r.score or 0.0) * (1.0 + boost * weight)), position, r, weight))
    keyed.sort(key=lambda item: (item[0], item[1]))
    out = []
    for _key, position, r, weight in keyed:
        if weight and len(out) < position:  # moved up because of its kind
            r.evidence_chain = [*(r.evidence_chain or []), f"kind_intent({r.fact.memory_kind})"]
        out.append(r)
    return out + tail


def _same_subject(a: Any, b: Any) -> bool:
    ea = set(getattr(a, "canonical_entities", None) or [])
    eb = set(getattr(b, "canonical_entities", None) or [])
    return bool(ea & eb)


def _latest_first(results: list, intent: Intent) -> list:
    """Within the head, a newer same-kind, same-subject memory precedes an older one."""
    wanted = {k.value for k in intent.kinds & _LATEST_KINDS}
    head, tail = list(results[:WINDOW]), results[WINDOW:]
    changed = True
    while changed:
        changed = False
        for i in range(len(head)):
            for j in range(i + 1, len(head)):
                older, newer = head[i].fact, head[j].fact
                if (older.memory_kind in wanted and newer.memory_kind == older.memory_kind
                        and _same_subject(older, newer)
                        and (newer.created_at or "") > (older.created_at or "")):
                    mover = head.pop(j)
                    head.insert(i, mover)
                    note = f"newer:{newer.fact_id}"
                    if note not in (head[i + 1].evidence_chain or []):
                        head[i + 1].evidence_chain = [*(head[i + 1].evidence_chain or []), note]
                    changed = True
                    break
            if changed:
                break
    return head + tail


def apply_kind_awareness(results: list, query: str, *, enabled: bool = True,
                         boost: float = DEFAULT_BOOST) -> list:
    """``results`` re-ordered for the question's intent; the same list object's
    items, nothing added or removed. Unchanged when disabled or no intent."""
    if not enabled or not results:
        return results
    intent = query_intent(query)
    if not intent.kinds:
        return results
    ordered = _boosted_order(list(results), intent, max(0.0, float(boost)))
    if intent.wants_latest:
        ordered = _latest_first(ordered, intent)
    for rank, r in enumerate(ordered, start=1):
        r.rank_position = rank
    return ordered


__all__ = ["DEFAULT_BOOST", "Intent", "WINDOW", "apply_kind_awareness", "query_intent"]
