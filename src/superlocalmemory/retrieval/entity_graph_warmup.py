# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's start-up build of the entity graph never makes a recall wait.

WHY
---
Measured on a copy of a 21,738-fact / 481,000-edge store: the first recall
after the daemon reported ready took 8.5-13 s, while its channels took 1.2 s.
About 6 s of the rest was spent queued on the entity graph's cache lock while
the start-up warm-up built that graph (edges, entity maps, metrics, the walk's
array view) -- the warm-up was doing the right work, and the person's recall
simply waited for it.

WHAT THIS DOES
--------------
* ``warm`` builds the graph for the active profile directly, first thing in the
  warm-up, under ``building``.
* The rest of the warm-up (its own recalls, which can rebuild a graph the
  enrichment keeps growing) runs under ``building`` too.
* ``score_candidates_unless_warming`` is what recall calls. While the start-up warm-up holds
  the graph, a recall does not queue behind it: it returns ``None``, which the
  engine reports as the ``entity_graph`` channel ``warming`` -- the answer is
  marked incomplete, never "found nothing" -- and returns everything else.

Outside the start-up build nothing changes: a recall that finds no cached graph
builds it itself, as before, because a one-shot command has no warm-up and
must not lose the boost on every call.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

#: id(channel) -> how many start-up warm-up scopes are open (they nest).
_building: dict[int, int] = {}
_building_lock = threading.Lock()


@contextmanager
def building(channel: Any) -> Iterator[None]:
    """Mark ``channel``'s graph as in use by the start-up warm-up."""
    key = id(channel)
    with _building_lock:
        _building[key] = _building.get(key, 0) + 1
    try:
        yield
    finally:
        with _building_lock:
            left = _building.get(key, 1) - 1
            if left > 0:
                _building[key] = left
            else:
                _building.pop(key, None)


def is_building(channel: Any) -> bool:
    with _building_lock:
        return id(channel) in _building


def _scope_key(channel: Any, profile_id: str, include_global: bool | None,
               include_shared: bool | None) -> tuple[str, bool, bool]:
    # Resolved exactly as EntityGraphChannel._score_candidates_locked does.
    if include_global is None:
        include_global = bool(getattr(channel, "include_global", False))
    if include_shared is None:
        include_shared = bool(getattr(channel, "include_shared", False))
    return (profile_id, bool(include_global), bool(include_shared))


def score_candidates_unless_warming(
    channel: Any, query: str, candidate_ids: list[str], profile_id: str, *,
    include_global: bool | None = None, include_shared: bool | None = None,
) -> dict[str, float] | None:
    """The channel's candidate scores, or None while the start-up build runs."""
    lock = getattr(channel, "_cache_lock", None)
    if lock is None or not is_building(channel):
        return channel.score_candidates(
            query, candidate_ids, profile_id,
            include_global=include_global, include_shared=include_shared)
    if not lock.acquire(blocking=False):
        return None  # the start-up build holds the graph: do not wait for it
    try:
        key = _scope_key(channel, profile_id, include_global, include_shared)
        if key not in getattr(channel, "_adj_slots", {}):
            # Between the build's steps, with nothing cached for this scope:
            # scoring now would start a second cold build on this recall.
            return None
        return channel._score_candidates_locked(
            query, candidate_ids, profile_id,
            include_global=include_global, include_shared=include_shared)
    finally:
        lock.release()


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
        with building(channel), lock:
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


__all__ = ["building", "channel_of", "is_building", "score_candidates_unless_warming", "warm"]
