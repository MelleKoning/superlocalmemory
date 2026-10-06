# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The entity graph is rebuilt next to recall, not inside it, when it only grew.

WHY
---
The entity-graph stage keeps the whole graph in memory and rebuilt it, on the
recall that noticed, whenever the edge or fact count moved. On a store that is
still being enriched the counts move all the time -- measured on a copy of a
21,739-fact / 481,000-edge store: 15 rebuilds during 60 recalls, each 1.5-13 s
on that recall's own clock (edges, entity maps, metrics, the walk's array
view). A memory saved a second ago cost every reader a full rebuild.

WHAT CHANGES
------------
When the graph only GREW since the cached copy was built (new facts, new or
re-weighted links), recall keeps answering from the cached copy and one
background rebuild per scope replaces it as soon as it is ready -- seconds
later. The cost: for those seconds, a new memory's links do not yet boost its
neighbours. It is still found by every other channel, and the boost is never
the only way to it.

When anything the cached copy could still SHOW has since been erased,
withheld, soft-deleted or moved to another scope, it is not served at all: the
recall rebuilds synchronously, exactly as before, so a removed memory never
shapes an answer through a stale graph. That is checked against the
trigger-maintained ``fact_search_changes`` log; a store without the log keeps
the old synchronous rebuild everywhere.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any

from superlocalmemory.storage import fact_search_changes as changes

logger = logging.getLogger(__name__)

_inflight: set[tuple[int, tuple]] = set()
_inflight_lock = threading.Lock()
#: id(channel), scope_key -> change-log head when that slot's load began.
_slot_seq: dict[tuple[int, tuple], int] = {}


def note_loaded(channel: Any, scope_key: tuple, seq: int) -> None:
    _slot_seq[(id(channel), scope_key)] = seq


def log_head(db: Any) -> int | None:
    try:
        return changes.log_bounds(db)[0]
    except Exception:  # noqa: BLE001 -- no log: no safe way to serve stale
        return None


def can_serve_stale(channel: Any, scope_key: tuple, slot: Any) -> bool:
    """Whether the cached slot only lacks additions (see module docstring)."""
    seq = _slot_seq.get((id(channel), scope_key))
    db = channel._db
    if seq is None:
        return False
    try:
        head, oldest = changes.log_bounds(db)
        if head < seq or (oldest is not None and oldest > seq + 1 and head > seq):
            return False
        changed = changes.changed_fact_ids(db, seq, head, "v")
        shown = [f for f in changed if f in slot.visible_fact_ids]
        if not shown:
            return True
        from superlocalmemory.retrieval.scope_policy import authorized_fact_ids

        profile_id, include_global, include_shared = scope_key
        still = authorized_fact_ids(db, shown, profile_id, include_global=include_global,
                                    include_shared=include_shared)
        return set(shown) <= set(still)
    except Exception:  # noqa: BLE001 -- cannot prove it: rebuild now
        return False


def refresh_in_background(channel: Any, scope_key: tuple, *, current_count: int,
                          current_fact_count: int, now: float) -> bool:
    """Start (at most one per scope) a rebuild that swaps in when ready."""
    key = (id(channel), scope_key)
    with _inflight_lock:
        if key in _inflight:
            return True
        _inflight.add(key)

    def run() -> None:
        try:
            seq = log_head(channel._db)
            shadow = copy.copy(channel)  # the loader writes only its own fields
            profile_id, include_global, include_shared = scope_key
            slot = shadow._load_adjacency_from_db(
                profile_id, include_global=include_global, include_shared=include_shared,
                current_count=current_count, current_fact_count=current_fact_count,
                now=now)
            with channel._cache_lock:
                channel._adj_slots[scope_key] = slot
                if seq is not None:
                    note_loaded(channel, scope_key, seq)
        except Exception as exc:  # noqa: BLE001 -- the next recall rebuilds instead
            logger.warning("background graph rebuild failed (%s)", type(exc).__name__)
        finally:
            with _inflight_lock:
                _inflight.discard(key)

    threading.Thread(target=run, name="slm-graph-refresh", daemon=True).start()
    return True


__all__ = ["can_serve_stale", "log_head", "note_loaded", "refresh_in_background"]
