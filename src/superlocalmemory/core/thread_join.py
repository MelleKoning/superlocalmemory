# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Bounded joins for pool threads an owner has just shut down.

``ThreadPoolExecutor.shutdown(wait=False)`` returns while its workers are
still finishing. An owner whose ``close()`` returns at that point hands its
caller threads that keep touching the database, numpy and native libraries
after the caller believes everything is released -- in the test suite, those
were the threads still running when an unrelated test later crashed. Waiting
forever (``wait=True``) is no better: a worker wedged in a model call would
hold shutdown hostage. So: wait, but never past one shared deadline.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: The whole join budget of one ``close()``, shared by every pool it owns.
CLOSE_JOIN_SECONDS = 1.0


def executor_threads(executor: object) -> list[threading.Thread]:
    """The worker threads of a ``ThreadPoolExecutor`` (empty for anything else).

    Read BEFORE ``shutdown``: the threads are what must be joined afterwards.
    """
    threads = getattr(executor, "_threads", None)
    if not threads:
        return []
    try:
        return [t for t in list(threads) if isinstance(t, threading.Thread)]
    except Exception:  # noqa: BLE001 - a foreign executor type
        return []


def join_threads(
    threads: Iterable[threading.Thread], *,
    budget_seconds: float = CLOSE_JOIN_SECONDS, owner: str = "",
) -> list[threading.Thread]:
    """Join ``threads`` within ``budget_seconds`` in total; return stragglers.

    Never joins the calling thread (a pool closing itself from one of its own
    workers). Stragglers are logged, not raised: a close must stay bounded and
    must not fail because a worker is slow to notice it was cancelled.
    """
    current = threading.current_thread()
    pending = [t for t in threads if t is not current]
    deadline = time.monotonic() + max(0.0, budget_seconds)
    for thread in pending:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    stragglers = [t for t in pending if t.is_alive()]
    if stragglers:
        logger.warning(
            "%s close: %d worker thread(s) still running after %.1fs: %s",
            owner or "pool", len(stragglers), budget_seconds,
            ", ".join(sorted(t.name for t in stragglers)),
        )
    return stragglers
