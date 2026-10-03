# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The one post-retrieval kind filter every surface shares (LLD/WP8 4.1.19).

``recall``, ``search`` and ``list_recent`` all accept a ``kind`` filter that
narrows on the DISPLAYED kind - ``kind_fields()["memory_kind"]`` - so a legacy
row filters by the kind it is mapped to, not by whatever raw value (or
nothing) sits in its ``memory_kind`` column. Because the filter runs AFTER
retrieval, the caller must fetch more than ``limit`` candidates first or a
filtered answer can come back short even when enough matches exist further
down the unfiltered list. ``overfetch_limit`` is that arithmetic; every
surface calls it with its own retrieval limit BEFORE running the retrieval,
then calls ``filter_items_by_kind`` on the result.

Pure, stdlib-only, no DB access: both halves are plain list/int operations
so MCP, CLI (direct engine reads) and HTTP (through the recall pipeline) can
all call the exact same code instead of three hand-written loops that could
drift.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: How much larger than the caller's limit the retrieval pool must be before
#: filtering, so a kind filter still has candidates to keep.
OVERFETCH_FACTOR = 3

#: However large ``limit`` is, never ask a retrieval step for more than this
#: many candidates - bounds the extra work (and latency) the filter can cost.
OVERFETCH_CAP = 100


def overfetch_limit(limit: int) -> int:
    """How many candidates to retrieve so a kind filter can still fill ``limit``.

    ``limit <= 0`` returns 0 — there is nothing to overfetch for a caller
    asking for zero results. Never raises on a negative value.
    """
    if limit <= 0:
        return 0
    return min(limit * OVERFETCH_FACTOR, OVERFETCH_CAP)


def filter_items_by_kind(
    items: Sequence[Mapping[str, Any]],
    kind: str | None,
    limit: int,
    *,
    kind_key: str = "memory_kind",
) -> list[Mapping[str, Any]]:
    """Keep items whose ``kind_key`` equals ``kind``, in order, capped at ``limit``.

    ``kind`` falsy (``None`` or ``""``) means "no filter": the first ``limit``
    items pass through unchanged — the common case, byte-identical to every
    surface's behaviour before this filter existed. ``items`` is expected to
    already carry the DISPLAYED kind (e.g. the output of
    ``storage.memory_kinds.kind_fields``) — this function compares values
    only, it never classifies or re-derives a kind itself.
    """
    if not kind:
        return list(items[:limit])
    kept: list[Mapping[str, Any]] = []
    for item in items:
        if item.get(kind_key) == kind:
            kept.append(item)
            if len(kept) >= limit:
                break
    return kept


__all__ = ["OVERFETCH_CAP", "OVERFETCH_FACTOR", "filter_items_by_kind", "overfetch_limit"]
