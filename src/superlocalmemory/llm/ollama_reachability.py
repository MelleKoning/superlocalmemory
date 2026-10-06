# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Bounded, cached reachability for a local Ollama server.

``LLMBackbone.is_available()`` reported True for provider ``"ollama"``
unconditionally: true that no API key is needed, but a different question
from "is anything listening at this URL right now." A saved Mode B pointed
at an Ollama that was never started (or has since stopped) got silent,
per-call extraction/consolidation failures with nothing telling the caller
the local model was the reason, and Mode B never degraded to the Mode A
fallback every caller of ``is_available()`` already has ready to use.

The check itself is cheap -- ``GET /api/tags`` lists installed models
without loading one -- but ``is_available()`` is called once per fact on
hot paths (extraction, consolidation, temporal validation, summarization),
so even a cheap check needs to not run on every single one of those calls.
A short TTL cache does that: the first call in a window pays for the probe,
every call inside the window reuses its answer.
"""

from __future__ import annotations

import threading
import time

#: Long enough to get an answer from a server that is actually up; short
#: enough that a host with nothing listening is not mistaken for a hang.
PROBE_TIMEOUT_S = 0.3
#: How long one answer is trusted before the next call pays for a fresh
#: probe. Bounded so an Ollama started mid-session is noticed reasonably
#: soon, without re-probing on every one of many per-fact calls in between.
CACHE_TTL_S = 15.0

_lock = threading.Lock()
_cache: dict[str, tuple[float, bool]] = {}


def ollama_reachable(
    base_url: str, *, timeout: float = PROBE_TIMEOUT_S, cache_ttl: float = CACHE_TTL_S,
) -> bool:
    """Is an Ollama-compatible server answering at ``base_url`` right now.

    Never raises: any connection problem (nothing listening, DNS failure,
    timeout) reads as "not reachable," the same as every other network
    failure path in this codebase.
    """
    base = base_url.rstrip("/")
    now = time.monotonic()
    with _lock:
        cached = _cache.get(base)
        if cached is not None and now - cached[0] < cache_ttl:
            return cached[1]
    reachable = _probe(base, timeout)
    with _lock:
        _cache[base] = (now, reachable)
    return reachable


def _probe(base: str, timeout: float) -> bool:
    try:
        import httpx

        response = httpx.get(f"{base}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def clear_cache() -> None:
    """Test-only: forget every cached reachability answer."""
    with _lock:
        _cache.clear()
