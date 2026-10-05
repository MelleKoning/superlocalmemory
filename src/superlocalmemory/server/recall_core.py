# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""The daemon's recall, after the request has been read and authorised.

Extracted from the ``GET /recall`` handler in ``server/unified_daemon.py``
(4.1.21) so that a saved view runs exactly the recall ``/recall`` runs — the
same thread-pool call, the same full-recall semaphore, the same budget and
keyword fallback when the budget is exceeded, the same serializer and the same
envelope. Two copies of this would drift; one is what makes "a view gives the
same answer as the question it stands for" true by construction.

What stays in the handler: reading and validating query parameters, the
permission check, the unknown-profile 404, the session id and the actor. What
lives here is everything from "the request is good" to "the response body".

``origin`` tags the recall for the Answer Check history (who asked:
a dashboard test, or a saved view run from the dashboard, the CLI or an agent).
It is entered on the executor thread, because a context variable set on the
event loop does not cross ``run_in_executor``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("superlocalmemory.server.recall_core")

# v3.4.53: Limit concurrent full (non-fast) recalls. Without this, N parallel
# full recalls spawn N threads → Ollama serialises, the reranker lock queues,
# and total wall time is N × one recall. Three gives parallelism without
# oversaturation. Shared by every caller of :func:`run_recall`.
RECALL_SEMAPHORE = asyncio.Semaphore(3)


def recall_budget_s() -> float:
    """Generous latency budget for a recall before the keyword fallback (v3.8.3).

    SLM's value is quality recall under heavy multi-agent load, so semantic
    recall is given ample time; the keyword fallback is a LAST-RESORT safety
    net for a genuine hang (e.g. a wedged embedder), not a speed cutoff. Tune
    with SLM_SEARCH_RECALL_TIMEOUT_S (shared with the dashboard search route).
    """
    try:
        v = float(os.environ.get("SLM_SEARCH_RECALL_TIMEOUT_S", ""))
        return v if v > 0 else 25.0
    except (TypeError, ValueError):
        return 25.0


def sanitize_json_text(text: str) -> str:
    """Strip control characters that break JSON serialization.

    Facts ingested from agent conversations can contain raw control characters
    that survive database round-trips but fail JSON encoding of the response.
    They are replaced with spaces so the byte length is preserved and
    truncation stays predictable. Only the ASCII control range is touched.
    """
    if not text:
        return text
    if all(c >= " " or c in "\n\r\t" for c in text):
        return text
    return "".join(c if c >= " " or c in "\n\r\t" else " " for c in text)


@dataclass(frozen=True, slots=True)
class RecallCall:
    """One recall, fully resolved: every value already validated and normalised."""

    query: str
    limit: int
    session_id: str
    agent_id: str
    fast: bool
    profile_id: str = ""
    include_global: bool | None = None
    include_shared: bool | None = None
    window: str = ""
    as_of: str = ""
    known_as_of: str = ""
    valid_at: str = ""
    include_unknown: bool = False
    facets: Any = None
    skip_answer_check: bool = False
    no_reorder: bool = False
    full: bool = False
    include_source: bool = False
    include_marker: bool = False
    origin: str = ""


def _engine_call(engine: Any, call: RecallCall) -> Any:
    """``engine.recall`` with the call's arguments, on the executor thread."""
    from superlocalmemory.core import answer_check_history as history
    from superlocalmemory.core.answer_check_scope import skip_answer_check

    with skip_answer_check() if call.skip_answer_check else nullcontext(), \
            history.origin(call.origin) if call.origin else nullcontext():
        return engine.recall(
            call.query, limit=call.limit, session_id=call.session_id,
            agent_id=call.agent_id, fast=call.fast,
            profile_id=call.profile_id or None,
            include_global=call.include_global, include_shared=call.include_shared,
            window=call.window or None, as_of=call.as_of or None,
            known_as_of=call.known_as_of or None, valid_at=call.valid_at or None,
            include_unknown=call.include_unknown,
            # Only when given: an engine stand-in need not know these.
            **({"facets": call.facets} if call.facets is not None else {}),
            **({"answer_check": "no_reorder"} if call.no_reorder else {}),
        )


def _envelope(engine: Any, call: RecallCall, response: Any, snapshot: Any) -> dict:
    from superlocalmemory.server.recall_serializer import (
        recall_response_metadata,
        serialize_recall_response,
    )

    memory_ids = list({r.fact.memory_id for r in response.results[:call.limit]
                       if r.fact.memory_id})
    memory_map = (
        engine._db.get_memory_content_batch(
            memory_ids, call.profile_id or engine.profile_id,
            include_global=True, include_shared=True,
        ) if memory_ids else {}
    )
    retrieval = getattr(engine._config, "retrieval", None)
    results, no_confident_match = serialize_recall_response(
        response, limit=call.limit,
        memory_map={k: sanitize_json_text(v) for k, v in memory_map.items()},
        per_fact_max=getattr(retrieval, "recall_per_fact_max_chars", 2400),
        total_max=getattr(retrieval, "recall_total_max_chars", 12000),
        # Markers only on session-bearing recalls: a marker can only buy a
        # learning signal when a pending outcome exists to settle.
        include_marker=call.include_marker,
        full=call.full, include_source=call.include_source,
    )
    for item in results:
        item["content"] = sanitize_json_text(item.get("content", ""))
    return {
        "ok": True,
        # The profile that actually served this recall: the routed profile when
        # one was named, else the active one. profile_generation describes the
        # global switch state, which per-request routing never moves.
        "profile": call.profile_id or snapshot.profile_id,
        "profile_generation": snapshot.generation,
        "query": call.query,
        "query_type": response.query_type,
        "result_count": len(results),
        "retrieval_time_ms": round(response.retrieval_time_ms, 1),
        "channel_weights": {k: round(v, 3)
                            for k, v in (response.channel_weights or {}).items()},
        "total_candidates": getattr(response, "total_candidates", 0),
        "results": results,
        "count": len(results),
        "no_confident_match": no_confident_match,
        **recall_response_metadata(response),
    }


async def run_recall(engine: Any, call: RecallCall, *, app_state: Any) -> dict:
    """Run one recall and return its response body. Raises on an engine failure.

    The caller has already authorised the request and checked the profile.
    """
    from superlocalmemory.core.recall_gate import begin_recall, end_recall
    from superlocalmemory.server.profile_runtime import get_profile_runtime
    from superlocalmemory.server.recall_fallback import recall_keyword_fallback

    # Marks a recall in flight so the pending materializer pauses.
    begin_recall()
    # Deep recalls are gated; fast recalls keep their bounded channels and skip
    # remote agentic verification, so they do not need the semaphore.
    if not call.fast:
        await RECALL_SEMAPHORE.acquire()
    try:
        # v3.8.3: bound the recall so callers never hang on a wedged embedder.
        # Poll the executor future (which cannot be cancelled) without blocking
        # the loop; only past the generous budget is the keyword answer served.
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _engine_call, engine, call)
        deadline = loop.time() + recall_budget_s()
        while not future.done() and loop.time() < deadline:
            await asyncio.sleep(0.05)
        snapshot = get_profile_runtime(app_state).snapshot
        if not future.done():
            future.add_done_callback(lambda f: (f.cancelled() or f.exception()))
            logger.warning("recall: semantic recall exceeded %.0fs budget for %r — "
                           "serving keyword fallback", recall_budget_s(), call.query[:80])
            # The fallback honours the same facets the primary path was given.
            return recall_keyword_fallback(
                engine, call.query, call.limit, profile_id=call.profile_id or None,
                profile=call.profile_id or snapshot.profile_id,
                profile_generation=snapshot.generation, facets=call.facets,
            )
        return _envelope(engine, call, future.result(), snapshot)
    finally:
        if not call.fast:
            RECALL_SEMAPHORE.release()
        end_recall()


__all__ = ["RECALL_SEMAPHORE", "RecallCall", "recall_budget_s", "run_recall",
           "sanitize_json_text"]
