# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon builds the entity graph at start; a recall never waits for it.

WHY
---
Measured on a copy of a 21,738-fact / 481,000-edge store: the first recall
after the daemon reported ready took 8.5-13 s, while its channels took 1.2 s.
About 6 s of the rest was spent queued on the entity graph's cache lock while
the start-up warm-up built that graph inside its first recall.

WHAT THIS DOES
--------------
* ``warm`` builds the graph for the active profile directly, first thing in the
  warm-up (before its own recalls).
* Builds run off the cache lock (retrieval/adjacency_rcu), so a recall whose
  scope has a usable cached graph always uses it.
* ``score_candidates_unless_warming`` is what recall calls: when the scope has
  no usable graph and another thread is building it, it returns ``None``, which
  the engine reports as the ``entity_graph`` channel ``warming`` -- the answer
  is marked incomplete, never "found nothing".
"""

from __future__ import annotations

import logging
import time
from typing import Any

from superlocalmemory.retrieval.adjacency_rcu import GraphWarming

logger = logging.getLogger(__name__)


def score_candidates_unless_warming(
    channel: Any, query: str, candidate_ids: list[str], profile_id: str, *,
    include_global: bool | None = None, include_shared: bool | None = None,
) -> dict[str, float] | None:
    """The channel's candidate scores, or None while its graph is being built."""
    try:
        return channel.score_candidates(
            query, candidate_ids, profile_id,
            include_global=include_global, include_shared=include_shared)
    except GraphWarming:
        return None


def channel_of(engine: Any) -> Any:
    """The engine's entity-graph channel, or None."""
    return getattr(getattr(engine, "_retrieval_engine", engine), "_entity", None)


def warm(engine: Any, profile_id: str) -> bool:
    """Build the entity graph for ``profile_id`` before anyone recalls."""
    channel = channel_of(engine)
    lock = getattr(channel, "_cache_lock", None)
    if channel is None or lock is None or not hasattr(channel, "_ensure_adjacency"):
        return False
    t0 = time.monotonic()
    try:
        with lock:  # released while the graph builds (adjacency_rcu)
            channel._ensure_adjacency(
                profile_id,
                include_global=bool(getattr(channel, "include_global", False)),
                include_shared=bool(getattr(channel, "include_shared", False)),
            )
    except Exception as exc:  # noqa: BLE001 -- a recall builds it instead
        logger.warning("entity graph warm-up failed (%s)", type(exc).__name__)
        return False
    logger.info("entity graph warmed for profile %s in %.0f ms",
                profile_id, (time.monotonic() - t0) * 1000.0)
    return True


__all__ = ["channel_of", "score_candidates_unless_warming", "warm"]
