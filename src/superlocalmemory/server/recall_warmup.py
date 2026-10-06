# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's start-up recalls: they warm caches, and nobody asked them.

v3.4.62: the first user query used to take 15-24 s because it read the graph
tables from disk; v3.8.5: the first FULL recall paid an 8-13 s cold cost for the
ranking model and graph metrics. These recalls pay both at boot instead.

Each holds its own short ``operation_nowait()`` lease, so a profile switch
issued at daemon start is never blocked behind them; a pending transition
preempts the remaining ones (they complete on the next boot).

They are the system's own recalls, so they never ask the answer check: no
verdict, nothing sent to a hosted provider, nothing billed, and no memory
leaves the machine because the daemon started (``skip_answer_check()``). The
on-device check's model is warmed separately — locally, without a recall.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from superlocalmemory.core.answer_check_scope import skip_answer_check

logger = logging.getLogger(__name__)

#: One to load the graph page cache, one to warm the reranker and producers.
WARMUP_QUERIES = ("memory recall performance", "context injection retrieval")
FULL_PATH_QUERY = "memory recall performance"


def run_warmup_recalls(engine: Any, profile_runtime: Any, *,
                       warm_spreading_activation: Callable[[Any, Any], Any]) -> None:
    """Fire the start-up recalls, then warm the on-device check's model."""
    with skip_answer_check():
        _recalls(engine, profile_runtime, warm_spreading_activation)
    warm_answer_check(engine)


def _recalls(engine: Any, profile_runtime: Any,
             warm_spreading_activation: Callable[[Any, Any], Any]) -> None:
    # The in-memory vector and kind indexes first (retrieval/kind_scope): built
    # here, the warm-up recalls below and the first real one find them ready
    # instead of waiting behind the build past the channel guard.
    from superlocalmemory.retrieval import kind_scope

    kind_scope.warm(engine, str(getattr(engine, "profile_id", "") or "default"))
    for query in WARMUP_QUERIES:
        with profile_runtime.operation_nowait() as snapshot:
            if snapshot is None:
                logger.debug("Recall warmup preempted by profile transition "
                             "— skipping remaining warmup queries")
                break
            # Short lease, so a profile switch can drain within 5 s.
            engine.recall(query, limit=5, fast=True)
    # The fast recalls skip spreading activation; warm that channel directly so
    # the first FULL recall is not cold.
    warm_spreading_activation(engine, profile_runtime)
    # The fast recalls do not exercise the full ranking path or the agentic
    # round, so one full recall loads those at boot, not on a user's query.
    try:
        with profile_runtime.operation_nowait() as snapshot:
            if snapshot is not None:
                engine.recall(FULL_PATH_QUERY, limit=5, fast=False)
    except Exception as exc:  # noqa: BLE001 — best-effort; never blocks readiness
        logger.debug("Full-path warmup skipped (non-fatal): %s", exc)


def warm_answer_check(engine: Any) -> bool:
    """Start loading the on-device check's model, without asking it anything.

    The check is built idle so a one-shot command never loads a model it will
    not use; the long-running daemon loads it here, at boot, so the first
    recall a person makes can be judged. A hosted check has nothing to warm.
    """
    retrieval = getattr(engine, "_retrieval_engine", None)
    judge = getattr(retrieval, "_sufficiency_judge", None) if retrieval else None
    start = getattr(judge, "start_warmup", None)
    if not callable(start):
        return False
    try:
        start()
    except Exception as exc:  # noqa: BLE001 — a judge never breaks start-up
        logger.debug("Answer check warm-up skipped: %s", type(exc).__name__)
        return False
    return True


__all__ = ["FULL_PATH_QUERY", "WARMUP_QUERIES", "run_warmup_recalls", "warm_answer_check"]
