# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The entity graph is built off the cache lock, so no recall waits on a build.

WHY
---
A synchronous graph build (first use of a scope, or a rebuild after a memory
the cached graph could still show was erased or withheld) ran while holding
``EntityGraphChannel._cache_lock`` -- 1.5-13 s on a 21,738-fact / 481,000-edge
store. Every other recall, including ones with a perfectly usable cached graph
for their own scope, queued behind it. On a just-started daemon that put ~6 s
of somebody else's build into the first recall a person made.

WHAT THIS DOES (read-copy-update)
---------------------------------
The build runs on a shallow copy of the channel with the lock RELEASED, and the
finished slot is swapped in under the lock. While it runs:

* a recall whose scope has a usable cached graph uses it (the lock is only held
  for scoring, never for a build);
* a recall whose scope has no usable graph -- none cached, or the cached one may
  still show a removed memory -- does not wait for another thread's build of
  that scope: ``GraphWarming`` is raised and recall reports the entity-graph
  channel ``warming`` (incomplete, never "found nothing").

A recall that finds no build in flight builds the graph itself, exactly as
before: a one-shot command has no warm-up to rely on. A removed memory is still
never shaped into an answer by a stale graph (retrieval/adjacency_refresh).
"""

from __future__ import annotations

import copy
import threading
from typing import Any

from superlocalmemory.retrieval import adjacency_refresh as _refresh


class GraphWarming(RuntimeError):
    """Another thread is building this scope's graph and none usable is cached."""


_inflight: set[tuple[int, tuple]] = set()
_inflight_lock = threading.Lock()


def build_in_flight(channel: Any, scope_key: tuple) -> bool:
    with _inflight_lock:
        return (id(channel), scope_key) in _inflight


def _claim(key: tuple[int, tuple]) -> bool:
    with _inflight_lock:
        if key in _inflight:
            return False
        _inflight.add(key)
        return True


def build_slot(channel: Any, scope_key: tuple, *, current_count: int,
               current_fact_count: int, now: float) -> Any:
    """Build ``scope_key``'s slot with the cache lock released; swap it in.

    Called by ``_ensure_adjacency`` with the cache lock held (any depth).
    Returns with the lock held again and the slot installed. Raises
    ``GraphWarming`` when another thread is already building this scope.
    """
    key = (id(channel), scope_key)
    if not _claim(key):
        raise GraphWarming(f"entity graph for {scope_key[0]!r} is being built")
    lock = channel._cache_lock
    owned = bool(getattr(lock, "_is_owned", lambda: False)())
    saved = lock._release_save() if owned else None
    try:
        seq = _refresh.log_head(channel._db)
        shadow = copy.copy(channel)  # the loader writes only its own fields
        profile_id, include_global, include_shared = scope_key
        slot = shadow._load_adjacency_from_db(
            profile_id, include_global=include_global, include_shared=include_shared,
            current_count=current_count, current_fact_count=current_fact_count, now=now)
    finally:
        if owned:
            lock._acquire_restore(saved)
        with _inflight_lock:
            _inflight.discard(key)
    # Lock held again: install. Replacing an existing key keeps its old LRU
    # position, so the caller's move_to_end also counts the reload as a use.
    channel._adj_slots[scope_key] = slot
    if seq is not None:
        _refresh.note_loaded(channel, scope_key, seq)
    return slot


__all__ = ["GraphWarming", "build_in_flight", "build_slot"]
