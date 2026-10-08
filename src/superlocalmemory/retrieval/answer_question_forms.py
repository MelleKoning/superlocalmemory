# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Two question shapes the on-device answer check misread, and what fixes them.

Measured on the pinned Laya weights (4.1.22): the check abstained on memories
that plainly answer two kinds of question.

* **Order** — "Which mobile platform was chosen first?" against "We decided to
  ship on iOS first and Android one quarter later" scored 0.38. The same
  question asked as "What is the order or sequence: ...?" scores 0.62 on that
  memory and stays low on memories that state no order.
* **Permission** — "Can we deploy X on a Friday?" against "Never deploy X on a
  Friday" scored 0.23: the model reads a yes/no question as needing a yes or a
  no, and a prohibition as neither. Asked without the generic subject ("Is it
  allowed to deploy X on a Friday?") it scores 0.81.

So for those shapes only, the check also asks the reworded question and keeps,
per memory, the higher of the two probabilities. Every other question is asked
exactly as before, with the same wording and the same threshold.

One case no wording fixed: "May the agent publish without approval?" against
"Never publish until the owner explicitly approves the release" (0.08). For a
permission question whose subject is generic (we, I, anyone, the agent ...),
an explicit rule recognises it: a sentence of the memory where a prohibition or
permission cue governs the asked action, and which carries every other content
word of the question. That is a rule, not a probability: the verdict lists the
memories it recognised (``rule_support``) and keeps the model's own numbers.

Never a lower threshold, and never applied to anything else: a known negative
scored above the approval case, so lowering it would accept that too.
Changing either part is a new ``FORMS_ID``; a verdict carrying either part
names it in its calibration id.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: Part of the calibration id of every verdict these forms touched.
FORMS_ID = "qforms-v1"

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
_ORDER = re.compile(
    r"\b(?:first|second|third|last|initially|earlier|later|before|after|"
    r"in what order|which order)\b", re.I)

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
_SENTENCE = re.compile(r"[^.!?;\n]+")
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


def is_order_question(question: str) -> bool:
    return (isinstance(question, str) and len(question) <= _MAX_QUESTION_CHARS
            and _ORDER.search(question) is not None)


def reworded(question: str) -> tuple[str, ...]:
    """The extra wordings the model is asked, in a fixed order (empty for most)."""
    if not isinstance(question, str):
        return ()
    body = _strip(question).rstrip("?").strip()
    out: list[str] = []
    action = permission_action(question)
    if action:
        out.append(f"Is it allowed to {action}?")
    if is_order_question(question):
        out.append(f"What is the order or sequence: {body}?")
    return tuple(q for q in out if q != question)


def applies(question: str) -> bool:
    return bool(reworded(question)) or permission_action(question) is not None


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _stem(word: str) -> str:
    return word if word.isdigit() else word[:_STEM]


def rule_supports(question: str, memory: str) -> bool:
    """Whether one memory states the rule that settles a permission question.

    All of: the question is a generic-subject permission question; one sentence
    of the memory has a prohibition/permission cue directly governing the asked
    action (first word of the action, matched on its first five letters); and
    that sentence carries every other content word of the question.
    """
    action = permission_action(question)
    if action is None or not isinstance(memory, str) or len(memory) > _MAX_RULE_CHARS:
        return False
    words = [w for w in _words(action) if w not in _STOP]
    if not words or words[0].isdigit():
        return False
    verb, rest = _stem(words[0]), {_stem(w) for w in words[1:]}
    governed = re.compile(_CUE + r"\s+(?:\w+\s+){0,2}?" + re.escape(verb) + r"\w*", re.I)
    for sentence in _SENTENCE.findall(memory):
        if not governed.search(sentence):
            continue
        stems = {_stem(w) for w in _words(sentence)}
        if rest <= stems:
            return True
    return False


def rule_support(question: str, memories: Sequence[str]) -> tuple[int, ...]:
    """Indices of the memories the permission rule recognises (usually none)."""
    if permission_action(question) is None:
        return ()
    return tuple(i for i, m in enumerate(memories) if rule_supports(question, m))


def merged(first: Sequence[float], second: Sequence[float]) -> tuple[float, ...]:
    """Per memory, the higher probability of two wordings."""
    return tuple(max(a, b) for a, b in zip(first, second, strict=True))


__all__ = ["FORMS_ID", "applies", "is_order_question", "merged", "permission_action",
           "reworded", "rule_support", "rule_supports"]
