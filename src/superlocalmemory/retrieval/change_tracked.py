# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The freshness rule shared by recall's in-memory, per-profile caches.

Before every read the cache applies the facts changed since its last read, as
named by the trigger-maintained ``fact_search_changes`` log, and nothing else.
It rebuilds a profile from the store when it cannot prove it is current: the
log went backwards (a restored or replaced store file) or the cache fell
further behind than the log keeps. A store without the log cannot prove
anything, so the cache reports itself unavailable and the caller keeps its
uncached path.

Why the order is safe: SQLite admits one writer at a time, so log sequence
numbers become visible in commit order. The head is read BEFORE the rows it
covers, so a write that lands in between is applied now and again next time --
idempotent -- and can never be skipped.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Generic, TypeVar

from superlocalmemory.storage import fact_search_changes as changes

logger = logging.getLogger(__name__)

P = TypeVar("P")


class ChangeTrackedCache(Generic[P]):
    """Per-profile partitions kept current from the change log."""

    #: ``v`` (vectors) or ``k`` (kinds): which logged changes this cache reads.
    WHAT = "v"

    def __init__(self, db: Any) -> None:
        self._db = db
        self._lock = threading.RLock()
        self._parts: dict[str, P] = {}
        self._seq = 0
        self._available: bool | None = None
        self.builds = 0
        self.deltas_applied = 0

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                changes.log_bounds(self._db)
                self._available = True
            except Exception as exc:  # noqa: BLE001 -- no log, no freshness proof
                logger.info("%s off (%s); the uncached path is kept",
                            type(self).__name__, exc)
                self._available = False
        return self._available

    def _fresh(self, profile_id: str) -> P:
        """The profile's partition, current as of now. Call with the lock held."""
        head, oldest = changes.log_bounds(self._db)
        if head < self._seq:
            self._parts.clear()
        elif head > self._seq and self._parts:
            if oldest is not None and oldest > self._seq + 1:
                self._parts.clear()
            else:
                ids = changes.changed_fact_ids(self._db, self._seq, head, self.WHAT)
                if ids:
                    self._apply(ids)
                    self.deltas_applied += len(ids)
        self._seq = head
        part = self._parts.get(profile_id)
        if part is None:
            part = self._build(profile_id)
            self._parts[profile_id] = part
            self.builds += 1
        return part

    def _build(self, profile_id: str) -> P:  # pragma: no cover - abstract
        raise NotImplementedError

    def _apply(self, fact_ids: list[str]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["ChangeTrackedCache"]
