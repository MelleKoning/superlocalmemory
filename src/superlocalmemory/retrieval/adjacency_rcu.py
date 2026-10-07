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

A recall that finds no build in flight builds the graph itself, as before: a
one-shot command has no warm-up to rely on. The long-running daemon instead
calls ``prefer_background_builds()``: there a build never runs on a recall's
clock -- it starts beside the recall, which reports ``warming`` -- because on a
just-started, loaded machine the cold build took 3-11 s of the first recalls.
A removed memory is still never shaped into an answer by a stale graph
(retrieval/adjacency_refresh).
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any

from superlocalmemory.retrieval import adjacency_refresh as _refresh

logger = logging.getLogger(__name__)


class GraphWarming(RuntimeError):
    """Another thread is building this scope's graph and none usable is cached."""


_inflight: set[tuple[int, tuple]] = set()
_inflight_lock = threading.Lock()
_background = False
_local = threading.local()


def prefer_background_builds(on: bool = True) -> None:
    """Daemon policy: build beside recall, never on its clock."""
    global _background
    _background = bool(on)


class building_here:
    """This thread is a builder (warm-up or background): it builds in place."""

    def __enter__(self):
        self._prev = getattr(_local, "builder", False)
        _local.builder = True
        return self

    def __exit__(self, *exc):
        _local.builder = self._prev
        return False


def _start_background(channel: Any, scope_key: tuple) -> None:
    if build_in_flight(channel, scope_key):
        return

    def run() -> None:
        profile_id, include_global, include_shared = scope_key
        try:
            with building_here(), channel._cache_lock:
                channel._ensure_adjacency(profile_id, include_global=include_global,
                                          include_shared=include_shared)
        except GraphWarming:
            pass  # another thread got there first
        except Exception as exc:  # noqa: BLE001 -- the next recall starts another
            logger.warning("background entity graph build failed (%s)", type(exc).__name__)

    threading.Thread(target=run, name="slm-graph-build", daemon=True).start()


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
    if _background and not getattr(_local, "builder", False):
        _start_background(channel, scope_key)
        raise GraphWarming(f"entity graph for {scope_key[0]!r} is being built")
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


__all__ = ["GraphWarming", "build_in_flight", "build_slot", "building_here",
           "prefer_background_builds"]
