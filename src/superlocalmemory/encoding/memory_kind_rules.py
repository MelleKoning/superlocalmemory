# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cue rules for the nine memory kinds — the suggestion that always runs and costs nothing.

First cue to match wins, in this priority (LLD §3.7):

    correction > rule > decision > procedure > prospective > status > opinion
    > episodic > the legacy kind of the fact's 4.1.18 type

What the numbers say, so nobody trusts this more than it deserves: the
research version scored 0.31 accuracy on 83 real, mostly long memories, 0.43
on memories under 300 characters and 0.17 over 800 (research §1, §6). So:

* Text over ``MAX_RULE_CHARS`` skips every cue and keeps its legacy kind — on
  long text a cue is more often an aside than the point.
* ``correction`` needs a strong cue (the memory itself says something was
  wrong), never a weak one like "turns out" or "supersedes".
* The answer is a suggestion (source ``rules``, no confidence). It never
  changes ``fact_type`` and never drives behaviour (LLD I2, I6).

Never raises, for any input: a kind must never be the reason a write fails (I1).
"""

from __future__ import annotations

import logging
import re

from superlocalmemory.encoding.memory_kind_recipe import RULES_RECIPE
from superlocalmemory.encoding.prospective_markers import looks_prospective
from superlocalmemory.storage.memory_kinds import (
    LEGACY_TO_KIND,
    KindAssignment,
    KindSource,
    MemoryKind,
)

logger = logging.getLogger(__name__)

#: Longer text keeps its legacy kind: rules scored 0.17 on text over 800 chars.
MAX_RULE_CHARS = 800

_I = re.IGNORECASE
_M = re.IGNORECASE | re.MULTILINE

#: Strong cues only: the memory itself says an earlier claim was wrong.
CORRECTION = re.compile(
    r"\bcorrection\s*(?::|to\b)|\berrat(?:um|a)\b|\bretract(?:s|ed|ion|ing)?\b|"
    r"\b(?:was|is|were|are) wrong\b|\bno longer true\b",
    _I,
)

_ACTION_VERBS = (r"use|run|do|commit|push|delete|call|route|dispatch|send|store|print|echo|"
                 r"write|merge|deploy|skip|add|set|claim|paste|edit|touch")

RULE = re.compile(
    r"\b(?:hard rule|non-negotiable|from now on|going forward|"
    r"never (?:" + _ACTION_VERBS + r")|"
    r"always (?:use|run|do|check|ask|verify|include|prefer|start|end|commit|write|read|keep)|"
    r"do not (?:" + _ACTION_VERBS + r")|"
    r"must (?:not |never |always )?[a-z]+|"
    r"(?:is|are) (?:banned|forbidden|not allowed|prohibited|mandatory|required)|"
    r"the rule is|rule:|policy:|convention:|"
    # A recurring obligation governs every future occurrence: a rule, not a plan.
    r"every (?:(?:mon|tues|wednes|thurs|fri|satur|sun)day|day|week|month|morning|evening|"
    r"night|sprint|quarter|release)\b)",
    _I,
)

DECISION = re.compile(
    r"\b(?:decided|decision(?: was|:)|we chose|i chose|chose to|chosen|opted (?:for|to)|"
    r"went with|settled on|final call|picked .{1,40} over|"
    r"go with|going with|selected .{1,40} over|in favou?r of|"
    r"approved|recommendation accepted|agreed to|ruled out)\b",
    _I,
)

PROCEDURE_LIST = re.compile(r"^\s*(?:\d+[.)]|step \d+)\s+\S", _M)
PROCEDURE = re.compile(
    r"(?:```|\$ \w|\b(?:how to|steps?:|to (?:install|run|deploy|release|build|configure|"
    r"set up|setup|fix|reproduce|upgrade|publish|test|start|restart|invoke|call)\b.{0,60}"
    r"\b(?:run|use|call|execute|type|open|set|add)\b|then run|first,? run|usage:|invoke with|"
    r"invocation:|command:|the command is))",
    _I,
)
#: "Heading: step, step, then step" — an ordered list written as one sentence —
#: or a named process heading ("Release flow:", "Runbook:").
PROCEDURE_SEQUENCE = re.compile(
    r"^[^:.\n]{2,40}:\s[^.\n]*,[^.\n]*\bthen\b|"
    r"\b(?:flow|workflow|process|procedure|routine|checklist|runbook|playbook):",
    _M,
)

PROSPECTIVE = re.compile(
    r"\b(?:todo|to-do|action items?|next steps?|will (?:draft|ship|implement|add|fix|write|send|"
    r"review|publish|release|run|create|build|update|merge)|plan(?:ned|ning)? to|"
    r"needs? to be (?:done|fixed|written|added)|follow[- ]up|remind me|still to do|"
    r"open items?|pending:)\b",
    _I,
)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
#: A deadline ("by 15 Dec", "by Friday", "by 5 pm") or a to-do written as an
#: imperative at the start of a clause ("— prepare the cost slide").
PROSPECTIVE_DEADLINE = re.compile(
    r"\bby (?:\d{1,2}(?:st|nd|rd|th)? " + _MONTH + r"|" + _MONTH + r" \d{1,2}|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|tomorrow|end of (?:day|week|month)|"
    r"eod|eow|\d{1,2}(?::\d\d)?\s*(?:am|pm))\b|"
    r"(?:^|[\u2014\u2013:;]\s*|\.\s+)(?:prepare|draft|send|book|schedule|email|submit|"
    r"remember to|follow up)\s+\w",
    _M,
)

STATUS = re.compile(
    r"\b(?:status(?: is|:)|in progress|currently|as of (?:now|today|\d)|so far|"
    r"(?:is|are) (?:blocked|running|pending|failing|passing|green|red|stuck|underway|ongoing)|"
    r"\d+\s*(?:/|of)\s*\d+\s+\w+\s+(?:passing|passed|done|complete|completed|filled|finished|"
    r"remaining)|"
    r"\d+\s*/\s*\d+\s+(?:tests?|passed|passing|done|complete)|work in progress|wip\b|"
    r"still (?:\w+ing|open|blocked)|remaining:|left to do)\b",
    _I,
)

OPINION = re.compile(
    r"\b(?:prefers?|preferred|preference|likes?|dislikes?|loves?|hates?|favou?rite|"
    r"i'd rather|would rather|i think|i believe|in my view|personally|"
    r"(?:better|worse) than|than it is worth|not worth it|overrated|underrated)\b",
    _I,
)

EPISODIC = re.compile(
    r"\b(?:yesterday|today|tonight|this morning|last (?:week|night|month|year|sprint)|"
    r"on (?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"released|shipped|merged|deployed|happened|occurred|met with|attended|visited|"
    r"broke|crashed|failed|passed|reached|launched|completed|finished|published|ran into|discovered|"
    r"found that|went|did|saw|we (?:fixed|found|built|added|removed|ran|tested))\b",
    _I,
)

#: (kind, test) in priority order; the first that fires wins.
_ORDER: tuple[tuple[MemoryKind, object], ...] = (
    (MemoryKind.CORRECTION, CORRECTION.search),
    (MemoryKind.RULE, RULE.search),
    (MemoryKind.DECISION, DECISION.search),
    (MemoryKind.PROCEDURE, lambda t: (len(PROCEDURE_LIST.findall(t)) >= 2
                                      or PROCEDURE.search(t) or PROCEDURE_SEQUENCE.search(t))),
    (MemoryKind.PROSPECTIVE, lambda t: (looks_prospective(t) or PROSPECTIVE.search(t)
                                        or PROSPECTIVE_DEADLINE.search(t))),
    (MemoryKind.STATUS, STATUS.search),
    (MemoryKind.OPINION, OPINION.search),
    (MemoryKind.EPISODIC, EPISODIC.search),
)


def legacy_kind(legacy_fact_type: object) -> MemoryKind:
    """The kind nearest a 4.1.18 ``fact_type`` value (or its enum); semantic if unknown."""
    value = getattr(legacy_fact_type, "value", legacy_fact_type)
    if not isinstance(value, str):
        return MemoryKind.SEMANTIC
    return LEGACY_TO_KIND.get(value.strip().lower(), MemoryKind.SEMANTIC)


def cue_kind(text: object) -> MemoryKind | None:
    """The first cue that fires on ``text``, or None (also for long or non-text input)."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_RULE_CHARS:
        return None
    for kind, test in _ORDER:
        if test(text):  # type: ignore[operator]
            return kind
    return None


def suggest_by_rules(text: str, legacy_fact_type: str) -> KindAssignment:
    """The rules suggestion for one fact. Never raises."""
    try:
        kind = cue_kind(text) or legacy_kind(legacy_fact_type)
    except Exception as exc:  # noqa: BLE001 — a cue bug must never fail a write
        logger.warning("memory kind rules failed (%s); using the legacy kind",
                       type(exc).__name__)
        kind = MemoryKind.SEMANTIC
        try:
            kind = legacy_kind(legacy_fact_type)
        except Exception:  # noqa: BLE001
            pass
    return KindAssignment(kind, KindSource.RULES, None, RULES_RECIPE)


__all__ = ["MAX_RULE_CHARS", "RULES_RECIPE", "cue_kind", "legacy_kind", "suggest_by_rules"]
