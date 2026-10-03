# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What counts as an exact lexical hit for a question - one definition.

Recall keeps the strongest such hit first, after every learned layer
(``core.recall_pipeline._preserve_exact_lexical_evidence``), and no later
re-ordering may move anything above it (``retrieval.kind_aware``). Both read
this module, so the guard and the passes that must respect it can never
disagree about which result it is.
"""

from __future__ import annotations

from typing import Any

#: A normalized question shorter than this is too generic to pin anything.
MIN_QUERY_CHARS = 3


def normalize_query(query: str) -> str:
    """Case-folded, whitespace-collapsed question text."""
    return " ".join(str(query or "").casefold().split())


def bm25_strength(result: Any) -> float:
    """The result's BM25 channel score (0.0 when it has none)."""
    return float((getattr(result, "channel_scores", None) or {}).get("bm25", 0.0) or 0.0)


def is_exact_lexical_hit(result: Any, normalized_query: str) -> bool:
    """True when ``result`` earned BM25 evidence and contains the whole question."""
    if len(normalized_query) < MIN_QUERY_CHARS or bm25_strength(result) <= 0.0:
        return False
    content = getattr(getattr(result, "fact", None), "content", "") or ""
    return normalized_query in " ".join(content.casefold().split())


__all__ = ["MIN_QUERY_CHARS", "bm25_strength", "is_exact_lexical_hit", "normalize_query"]
