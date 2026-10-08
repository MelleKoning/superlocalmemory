# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A permission question answered by a stated rule, which the model misreads.

Measured on the pinned Laya weights (4.1.22): asked "May the agent publish
without approval?", the on-device answer check scored the memory "Never publish
until the owner explicitly approves the release" at 0.08 and abstained. The
model reads a yes/no question as needing a yes or a no, and a prohibition as
neither. No threshold can fix that: a known negative scored 0.22, above it.

So, for a yes/no permission question whose subject is generic (we, I, anyone,
the agent ...), an explicit rule also reads the memories: a memory settles the
question when one of its sentences has a prohibition or permission cue that
governs the asked action ("never publish", "must not be booked") AND carries
every other content word of the question. That is a rule, not a probability:
the verdict lists the memories it recognised (``rule_support``) and keeps the
model's own numbers untouched.

Tried and rejected on data (4.1.22): also asking the model a reworded question
("Is it allowed to ...?", "What is the order or sequence: ...?") fixed order
and permission cases on the tuning set, but on a blind held-out set it added
two false accepts (a rule about another product, a pair whose order was
undecided) for one fix. The rule alone added none.

Every other question is judged exactly as before. Changing the rule is a new
``FORMS_ID``.

Calibration identity: the rule's id is appended (``RULE_SUFFIX``) to a verdict's
calibration id ONLY when the rule settled it (``rule_support`` non-empty). When
the rule read a question and recognised nothing, the decision is the model's
probabilities against the model's threshold — the very function the base id
names, giving the identical verdict — so the base id is the true identity, and
a verdict that is the same as before carries the same name as before. The
suffix therefore marks exactly the verdicts a rule decided, which is also how
every surface tells them apart (``calibration_id`` already reaches all of them;
``answer_check_reason`` cannot carry it, since it says "reused" for a verdict
repeated from memory).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: Names the rule in the calibration id of every verdict it settled.
FORMS_ID = "permission-rule-v1"
RULE_SUFFIX = "+" + FORMS_ID

_GENERIC = (r"(?:we|i|you|they|anyone|anybody|someone|somebody|one|"
            r"the (?:agent|assistant|ai|bot|team|user))")
_MODAL_GENERIC = re.compile(
    r"^\s*(?:may|can|could|should)\s+" + _GENERIC + r"\s+(?P<rest>[^?]+?)\s*\??\s*$", re.I)
_ALLOWED_GENERIC = re.compile(
    r"^\s*(?:is|are|am)\s+" + _GENERIC + r"\s+(?:allowed|permitted)\s+to\s+"
    r"(?P<rest>[^?]+?)\s*\??\s*$", re.I)
_ALLOWED_IT = re.compile(
    r"^\s*is\s+it\s+(?:ok|okay|fine|allowed|permitted)\s+(?:for\s+" + _GENERIC + r"\s+)?to\s+"
    r"(?P<rest>[^?]+?)\s*\??\s*$", re.I)
#: A memory longer than this is not read by the rule (the model still is).
_MAX_RULE_CHARS = 600
_MAX_QUESTION_CHARS = 300
#: Prohibition / permission cues that must GOVERN the asked action: the action
#: word follows within two words ("never publish", "may only push",
#: "must not be deployed").
_CUE = (r"(?:never|do\s+not|don't|must\s+not|mustn't|may\s+not|cannot|can't|"
        r"should\s+not|shouldn't|not\s+allowed\s+to|not\s+permitted\s+to|"
        r"forbidden\s+to|prohibited\s+from|allowed\s+to|permitted\s+to|"
        r"no\s+one\s+may|nobody\s+may|may\s+only|must\s+only|only)")
_STOP = frozenset((
    "a an the to of on in at for by with without from into onto about over under "
    "and or but if then than so as is are be been being am was were do does did "
    "it its this that these those there here any some all just ever also still yet "
    "me my our your their his her them us him she he i we you they it's "
    "please now today tonight").split())
#: A sentence with the terminator that ended it ("?" marks a question).
_SENTENCE = re.compile(r"[^.!?;\n]+[.!?;\n]?")
#: A sentence that opens like a question even without its "?": an auxiliary or
#: modal followed by a subject ("Should we ...", "Do we ...", "Is it ...").
#: "Do not publish" is an imperative, not a question: "not" is no subject.
_INTERROGATIVE = re.compile(
    r"^\W*(?:may|can|could|should|shall|would|will|must|is|are|am|was|were|do|does|did|"
    r"has|have|had)\s+(?:" + _GENERIC[3:-1] + r"|it|he|she|there|this|that|"
    r"(?:the|a|an|our|your|their|this|that)\s+\w+)\b", re.I)
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")
_STEM = 5


def _strip(question: str) -> str:
    return " ".join(question.split())


def permission_action(question: str) -> str | None:
    """The asked action of a yes/no permission question with a generic subject.

    "May the agent publish without approval?" -> "publish without approval".
    None for anything else, including a question naming a specific subject:
    that subject is part of what must be matched, and is left to the model.
    """
    if not isinstance(question, str) or len(question) > _MAX_QUESTION_CHARS:
        return None
    text = _strip(question)
    for pattern in (_MODAL_GENERIC, _ALLOWED_GENERIC, _ALLOWED_IT):
        found = pattern.match(text)
        if found:
            return found["rest"]
    return None


def applies(question: str) -> bool:
    """Whether the rule reads this question at all (else: model only, as before)."""
    return permission_action(question) is not None


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _stem(word: str) -> str:
    return word if word.isdigit() else word[:_STEM]


def _is_question(sentence: str) -> bool:
    """A memory sentence that asks rather than states ("Should we never ...?")."""
    return sentence.rstrip().endswith("?") or _INTERROGATIVE.match(sentence) is not None


def rule_supports(question: str, memory: str) -> bool:
    """Whether one memory states the rule that settles a permission question.

    All of: the question is a generic-subject permission question; one sentence
    of the memory has a prohibition/permission cue directly governing the asked
    action (first word of the action, matched on its first five letters); that
    sentence carries every other content word of the question; and it states
    rather than asks (no "?", no question opening like "Should we ...").
    """
    action = permission_action(question)
    if action is None or not isinstance(memory, str) or len(memory) > _MAX_RULE_CHARS:
        return False
    words = [w for w in _words(action) if w not in _STOP]
    if not words or words[0].isdigit():
        return False
    verb, rest = _stem(words[0]), {_stem(w) for w in words[1:]}
    governed = re.compile(_CUE + r"\s+(?:\w+\s+){0,2}?" + re.escape(verb) + r"\w*", re.I)
    asked = rest | {verb}
    for sentence in _SENTENCE.findall(memory):
        found = None if _is_question(sentence) else governed.search(sentence)
        if found is None:
            continue
        stems = {_stem(w) for w in _words(sentence)}
        if rest <= stems and _object_stems(found.group(0), sentence[found.end():]) <= asked:
            return True
    return False


#: Words that open what follows the action without naming what it acts on.
_OBJECT_END = _STOP | frozenset((
    "until unless before after when whenever while except during because since till once "
    "again first anything something everything nothing anyone anybody everyone").split())
_OBJECT_LEAD = frozenset("the a an any this that these those our your their my".split())
_OBJECT_MAX = 4


def _object_stems(governing: str, tail: str) -> set[str]:
    """Stems of what the governed action acts on ("publish Atlas ..." -> {"atlas"}).

    The words right after the action, skipping a leading article, up to the first
    punctuation or word that starts a condition or a prepositional phrase. A rule
    about one thing ("Never publish Atlas without approval") answers only a
    question about that thing; "Never publish until the owner approves" acts on
    the action itself. A passive action ("must never be deleted") has no object.
    """
    if re.search(r"\bbe(?:en)?\b", governing, re.I):
        return set()
    words = _words(re.split(r"[,;:(\[]", tail, maxsplit=1)[0])
    while words and words[0] in _OBJECT_LEAD:
        words = words[1:]
    out: set[str] = set()
    for word in words[:_OBJECT_MAX]:
        if word in _OBJECT_END:
            break
        out.add(_stem(word))
    return out


def rule_support(question: str, memories: Sequence[str]) -> tuple[int, ...]:
    """Indices of the memories the permission rule recognises (usually none)."""
    if permission_action(question) is None:
        return ()
    return tuple(i for i, m in enumerate(memories) if rule_supports(question, m))


def calibration_id_for(base_id: str, support: Sequence[int]) -> str:
    """The verdict's calibration id: the rule's suffix only when the rule settled it."""
    return base_id + RULE_SUFFIX if support else base_id


def base_calibration_id(calibration_id: str) -> str:
    """The judge's own id, with the rule's suffix (if any) set aside."""
    if settled_by_rule(calibration_id):
        return calibration_id[: -len(RULE_SUFFIX)]
    return calibration_id


def settled_by_rule(calibration_id: object) -> bool:
    """Whether a verdict (by its calibration id, on any surface) was settled by the rule."""
    return isinstance(calibration_id, str) and calibration_id.endswith(RULE_SUFFIX)


__all__ = ["FORMS_ID", "RULE_SUFFIX", "applies", "base_calibration_id", "calibration_id_for",
           "permission_action", "rule_support", "rule_supports", "settled_by_rule"]
