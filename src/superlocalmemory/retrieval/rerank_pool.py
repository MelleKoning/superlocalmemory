# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which candidates the cross-encoder reads - one definition."""

from __future__ import annotations

from collections.abc import Iterable, Set

from superlocalmemory.retrieval.fusion import FusionResult


def rerank_pool(
    fused: Iterable[FusionResult], *, found: Set[str], size: int,
) -> list[FusionResult]:
    """The best ``size`` search results and the best ``size`` added neighbours.

    WHY. After fusion, recall adds neighbours of its best results: the other
    memories in their scenes and bridge memories between them. Each gets a
    score taken from the result it came from (a scene sibling 0.8x its
    seed's), and a bridge gets a fixed 0.4 to 0.8, while a fused score is a
    sum of ``w / (k + rank)`` terms that rarely passes 0.3. One long session's
    scene therefore placed dozens of siblings above every result but the first
    few. Cutting that list at ``size`` gave the cross-encoder a pool that was
    86% added neighbours on the owner's eval, and a memory a channel had
    found, ranked 14th, never reached the cross-encoder at all.

    Giving each kind its own ``size`` slots means an added neighbour can still
    reach the cross-encoder, but cannot take a search result's place. The pool
    is in fused order and contains the plain ``fused[:size]`` cut as its
    prefix, so with no cross-encoder the order recall returns is unchanged.
    """
    ranked = list(fused)
    keep_found = [fr.fact_id for fr in ranked if fr.fact_id in found][:size]
    keep_added = [fr.fact_id for fr in ranked if fr.fact_id not in found][:size]
    keep = set(keep_found) | set(keep_added)
    return [fr for fr in ranked if fr.fact_id in keep]


__all__ = ["rerank_pool"]
