# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Dotted version numbers as whole search terms - one definition.

WHY. ``atomic_facts_fts`` uses FTS5's default ``unicode61`` tokenizer, which
treats ``.`` as a separator: ``4.1.20`` is indexed as the three words ``4``,
``1`` and ``20``. Every keyword path then tokenized the question the same way
and OR-joined the pieces, so a question about 4.1.20 asked "any memory with a
4, a 1 or a 20". On a live store those numbers are in times, counts and dates
everywhere; FTS5 gives a word found in more than half the rows an IDF of
~1e-6, so every 4.1.x memory scored the same and the order was an accident.

WHY NOT RE-TOKENIZE THE INDEX. The index already holds what is needed. FTS5
stores token positions (``detail=full``, the default, and no table here sets
anything else), so ``4.1.20`` is recorded as ``4`` ``1`` ``20`` at adjacent
positions. A quoted FTS5 string is a phrase query, and ``"4.1.20"`` is
tokenized into exactly that adjacent run. It is an exact version match on the
existing index, scored by FTS5's own BM25 with its own IDF for the phrase.
Nothing has to be rebuilt, so no store needs a migration or a restore point.

What a phrase alone cannot see is the token after it: ``"4.1.20"`` also
matches inside ``4.1.20.1`` and ``3.4.1.20``. ``version_terms`` reads the
whole dotted run from text, so callers can tell an exact mention from one of
those longer numbers. ``retrieval.bm25_channel`` uses it to take the phrase's
BM25 share back from such rows.

The single words stay in the question as well, so a partial version (``4.1``
against a memory about ``4.1.20``) still matches through them.
"""

from __future__ import annotations

import re

# A maximal dotted run of digits, optionally written with a leading v.
# The lookbehind refuses to start inside a word or another dotted run
# ("dev4.1", "1.4.1"); the lookahead refuses to stop early ("4.1.20rc1"),
# while a sentence-ending full stop ("shipped 4.1.20.") still ends the run.
_VERSION_RE = re.compile(r"(?<![\w.])[vV]?([0-9]+(?:\.[0-9]+)+)(?!\.?\w)")


def version_terms(text: str) -> tuple[str, ...]:
    """Distinct dotted version numbers in ``text``, without a leading v.

    First-seen order, so the same text always yields the same terms.
    """
    seen: dict[str, None] = {}
    for match in _VERSION_RE.finditer(text or ""):
        seen.setdefault(match.group(1), None)
    return tuple(seen)


def version_match_phrases(text: str) -> tuple[str, ...]:
    """FTS5 MATCH phrases for every version in ``text``.

    Each version is offered as written and with a leading ``v``: the
    tokenizer keeps ``v4`` as one word, so ``v4.1.20`` in a memory is the
    run ``v4`` ``1`` ``20`` and only the second phrase reaches it. The
    phrases hold digits, dots and ``v`` only, so they are always valid
    MATCH syntax.
    """
    phrases: list[str] = []
    for term in version_terms(text):
        phrases.extend((f'"{term}"', f'"v{term}"'))
    return tuple(phrases)


__all__ = ["version_match_phrases", "version_terms"]
