# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Embed a recall query without letting a loading model hold the whole recall.

WHY THIS EXISTS
---------------
On a fresh daemon the local embedding model takes tens of seconds to load (a
measured 32.7 s on a fresh install). The query embedding used to be a plain
blocking call made BEFORE any channel was dispatched, so every recall in that
window sat behind the model load — keyword search included, though it needs no
embedding at all. The daemon's last-resort budget then fired and answered with
a degraded fallback, and a memory saved seconds earlier with a "queryable"
receipt came back as "No confident match".

The embedding now runs on its own worker and the recall waits for it no longer
than the per-channel hang guard (``CHANNEL_HANG_GUARD_SECONDS``) — the limit
every channel is already held to, so this adds no new cutoff to the embedding
channels' work, it only moves the clock to include the step that feeds them. If
the vector is not back by then, the channels that need it are reported as
``warming`` (model still loading) or ``timeout`` (model loaded but slow) and the
recall is marked incomplete, while the channels that need no vector run and can
find the memory.

WHAT IT COSTS IN QUALITY
------------------------
Nothing on a warm daemon: an embed there takes well under a second, far inside
the guard. In the cold window the alternative was not "a recall with semantic
search", it was a recall with NO channels at all (the daemon budget fallback),
so this strictly adds results. The abandoned embed is not cancelled: it finishes
on its worker and lands in the query cache, so the same question asked again
gets the full set of channels as soon as the model is up.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Callable

from superlocalmemory.retrieval import channel_status as chstat

logger = logging.getLogger(__name__)

__all__ = ["QueryEmbedder"]


class QueryEmbedder:
    """Bounded, single-flight, cached query embedding for one retrieval engine.

    ``embed`` returns ``(vector, status)``. ``status`` is ``None`` when the
    embedder answered in time (the vector may still be ``None`` if the embedder
    itself returned nothing — that stays the caller's ``no_embedding`` case),
    and ``WARMING`` / ``TIMEOUT`` when the recall stopped waiting.
    """

    def __init__(
        self, embedder: Callable[[], Any], *, cache_max_size: int = 512,
    ) -> None:
        # A provider, not the object: the owning engine's embedder can be
        # swapped after construction and every call must see the current one.
        self._provider = embedder
        self._cache: dict[str, list[float]] = {}
        self._cache_max_size = cache_max_size
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future] = {}
        # Created on first use: an engine with no embedder (Mode A without
        # vectors) or one only ever used from background work owns no threads.
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._closed = False

    @property
    def cache(self) -> dict[str, list[float]]:
        return self._cache

    def _remember(self, query: str, vector: list[float] | None) -> None:
        if vector is None:
            return
        with self._lock:
            if query not in self._cache and len(self._cache) >= self._cache_max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[query] = vector

    def _compute(self, query: str) -> list[float] | None:
        vector = self._provider().embed(query)
        self._remember(query, vector)
        return vector

    def _future_for(self, query: str) -> concurrent.futures.Future:
        with self._lock:
            fut = self._inflight.get(query)
            if fut is not None:
                return fut
            if self._executor is None:
                if self._closed:
                    raise RuntimeError("query embedder is closed")
                # Two workers: one may be parked behind a cold model load
                # while a second question still gets its turn once it is up.
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="slm-query-embed",
                )
            fut = self._executor.submit(self._compute, query)
            self._inflight[query] = fut

        def _done(_f, q=query) -> None:
            with self._lock:
                if self._inflight.get(q) is _f:
                    del self._inflight[q]

        # Outside the lock: a future that is already done runs the callback
        # inline, and the callback takes the same (non-reentrant) lock.
        fut.add_done_callback(_done)
        return fut

    def _abandoned_status(self) -> str:
        """``warming`` when the embedder says its model is not loaded yet."""
        warm = getattr(self._provider(), "is_warm", None)
        return chstat.WARMING if warm is False else chstat.TIMEOUT

    def embed(self, query: str, wait_seconds: float) -> tuple[list[float] | None, str | None]:
        """Embed ``query``, waiting at most ``wait_seconds`` for the vector.

        Raises whatever the embedder raised when it answered in time with an
        error, exactly as the direct call did.
        """
        if self._provider() is None:
            return None, None
        cached = self._cache.get(query)
        if cached is not None:
            return cached, None
        from superlocalmemory.core.recall_gate import is_background_work

        if is_background_work():
            # Background callers (materializer, warm-up) keep the exact old
            # behaviour: their thread-local priority marker does not cross to
            # a worker thread, and nobody is waiting on them interactively.
            return self._compute(query), None
        fut = self._future_for(query)
        try:
            return fut.result(timeout=max(0.0, wait_seconds)), None
        except concurrent.futures.TimeoutError:
            status = self._abandoned_status()
            logger.warning(
                "Query embedding not ready within %.1fs (%s); this recall runs "
                "without the channels that need it and is marked incomplete",
                wait_seconds, status,
            )
            return None, status

    def close(self) -> None:
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
