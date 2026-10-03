# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The ``kinds`` request to the Laya worker, and strict reading of its reply.

Kept out of ``sufficiency.py`` so the answer check's own module only gains the
locking it needs. Pure functions: no process, no lock, no I/O.

A reply is all-or-nothing: one answer per document, every choice one of the
recipe's labels, every number a finite probability. Anything else is None and
the caller falls back to the rules — a half-parsed reply would pair answers
with the wrong memories.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from superlocalmemory.encoding.memory_kind_recipe import KindAnswer, KindRecipe

#: Documents per worker request. The worker answers ~55 ms a document; two per
#: request means a recall that arrives mid-typing waits about a tenth of a
#: second for the worker, not the whole memory's worth.
CHUNK_DOCUMENTS = 2
#: The worker's own limit on one ``kinds`` request.
MAX_DOCUMENTS = 16


def chunk_timeout_s(n_documents: int) -> float:
    """LLD §3.4: ``min(3.0, 0.25 + 0.12 * n)``."""
    return min(3.0, 0.25 + 0.12 * max(0, n_documents))


def build_chunks(documents: Sequence[str], recipe: KindRecipe,
                 verify_indices: Sequence[int]) -> list[dict[str, Any]] | None:
    """The ``kinds`` requests for ``documents``, ``CHUNK_DOCUMENTS`` at a time.

    None for anything the worker would refuse: more than 16 documents, a
    document that is not text, a verify index out of range.
    """
    docs = list(documents)
    if len(docs) > MAX_DOCUMENTS or not all(isinstance(d, str) for d in docs):
        return None
    verify = set()
    for i in verify_indices:
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(docs):
            return None
        verify.add(i)
    chunks = []
    for start in range(0, len(docs), CHUNK_DOCUMENTS):
        part = docs[start:start + CHUNK_DOCUMENTS]
        req: dict[str, Any] = {
            "cmd": "kinds",
            "documents": [d[:recipe.max_chars] for d in part],
            "instructions": recipe.instructions,
            "criteria": dict(recipe.criteria),
        }
        local = [i - start for i in sorted(verify) if start <= i < start + len(part)]
        if local:
            req["verify"] = {"indices": local, "instructions": recipe.correction_verify}
        chunks.append(req)
    return chunks


def _probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) and 0.0 <= f <= 1.0 else None


def parse_answer(raw: object, recipe: KindRecipe, *, verify_expected: bool) -> KindAnswer | None:
    """One validated answer, or None."""
    if not isinstance(raw, Mapping):
        return None
    choice = raw.get("choice")
    if not isinstance(choice, str) or choice not in recipe.criteria:
        return None
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, Mapping) or not probabilities:
        return None
    clean: dict[str, float] = {}
    for label, value in probabilities.items():
        p = _probability(value)
        if label not in recipe.criteria or p is None:
            return None
        clean[label] = p
    confidence = _probability(raw.get("confidence"))
    if confidence is None:
        return None
    verify_raw = raw.get("verify")
    verify = _probability(verify_raw) if verify_raw is not None else None
    if verify_raw is not None and verify is None:
        return None
    if verify is not None and not verify_expected:
        return None
    return KindAnswer(choice=choice, probabilities=clean, confidence=confidence,
                      verify=verify)


def parse_reply(reply: object, request: Mapping[str, Any],
                recipe: KindRecipe) -> list[KindAnswer] | None:
    """All answers for one chunk, or None if any is missing or malformed."""
    if not isinstance(reply, Mapping) or reply.get("ok") is not True:
        return None
    answers = reply.get("answers")
    documents = request.get("documents") or []
    if not isinstance(answers, list) or len(answers) != len(documents):
        return None
    verify_at = set((request.get("verify") or {}).get("indices", []))
    out = []
    for i, raw in enumerate(answers):
        answer = parse_answer(raw, recipe, verify_expected=i in verify_at)
        if answer is None:
            return None
        out.append(answer)
    return out


__all__ = ["CHUNK_DOCUMENTS", "MAX_DOCUMENTS", "build_chunks", "chunk_timeout_s",
           "parse_answer", "parse_reply"]
