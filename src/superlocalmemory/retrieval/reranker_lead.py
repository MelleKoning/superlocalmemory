# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The cross-encoder's clear leads survive every layer that ranks after it.

WHY. After the cross-encoder scores the pool, recall still multiplies each
result by recency (0.8-1.1x), content length (0.3-1x), trust (0.5-1.5x) and,
for time questions, a recency prior (up to 1.5x), and then the learned layers
re-score the list again. Together they could move a memory by up to 20x, so on
the owner's eval a memory the cross-encoder put first was pushed below one it
scored lower, and the answer went from right to wrong on eight questions. Each
layer is a fair tie-breaker; none of them read the question, so none should
overturn the one stage that did.

THE RULE. The order the later layers chose is kept, except that a memory may
not stand above another whose reranker score exceeds its own by more than
``LEAD_BOUND``. Between memories within that factor of each other the later
layers decide, as before; beyond it the reranker does. The rule reads only the
reranker score and the order, so it holds whatever units a learned layer's
scores are in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

#: The largest reranker-score ratio the later layers may overturn.
#:
#: Measured on the owner's store over the 50 dev and 97 held-out questions,
#: with this rule off: the later layers overturned the reranker's first result
#: on 42 questions. Where that decided right or wrong, the layers were right
#: at leads of 1.001 and 1.026, and wrong at 1.014, 1.031, 1.033, 1.068, 1.077,
#: 1.096, 1.137 and 1.201. Below about 1.05 the two are mixed; above it the
#: reranker was right every time. So the layers keep ties within 5%, which is
#: also the size of the recency and trust differences between two memories of
#: similar age and standing, and lose everything larger. No single factor
#: separates every case: at 1.05 the 1.014, 1.031 and 1.033 ones stay wrong.
LEAD_BOUND: float = 1.05


def _anchor(result: Any, anchors: Mapping[str, float]) -> float | None:
    fact = getattr(result, "fact", None)
    value = anchors.get(getattr(fact, "fact_id", None)) if fact is not None else None
    return None if value is None else float(value)


def hold_reranker_lead(
    results: Sequence[Any], anchors: Mapping[str, float],
    *, bound: float = LEAD_BOUND,
) -> list[Any]:
    """``results`` reordered as little as needed to keep every clear lead.

    Walks the given order and, at each position, places the first remaining
    result that no remaining one leads by more than ``bound``. The remaining
    result with the highest reranker score is never blocked, so the walk
    always advances. Ranking scores stay with positions (the set of scores is
    unchanged and still falls with the order), so a later bounded pass that
    compares them sees the order it is given. A result without a reranker
    score - the reranker did not run, or a layer built results of its own -
    leaves the list untouched. New objects; the input is not modified.
    """
    items = list(results)
    if len(items) < 2 or bound <= 1.0:
        return items
    scores = [_anchor(r, anchors) for r in items]
    if any(s is None for s in scores):
        return items
    remaining = list(range(len(items)))
    order: list[int] = []
    while remaining:
        top = max(scores[i] for i in remaining)
        pick = next(i for i in remaining if top <= bound * scores[i])
        order.append(pick)
        remaining.remove(pick)
    if order == list(range(len(items))):
        return items
    keys = [getattr(r, "ranking_score", None) for r in items]
    out = []
    for position, index in enumerate(order):
        result = items[index]
        if all(k is not None for k in keys):
            result = replace(result, ranking_score=sorted(keys, reverse=True)[position])
        out.append(result)
    return out


__all__ = ["LEAD_BOUND", "hold_reranker_lead"]
