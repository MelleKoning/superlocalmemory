# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""What ``slm warmup`` may claim about a running daemon.

``slm warmup`` used to print PASS as soon as the daemon reported an engine
object, while the same /health response said ``ready: false`` and
``embedding_warm: false``. The first recall after that PASS then paid the
whole model load. PASS now means what it says: the daemon reports itself
ready AND the embedding model reports itself warm. Anything short of that is
waited on up to a deadline and, if it never arrives, reported as not ready
with a non-zero exit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

#: How long ``slm warmup`` waits for a starting daemon by default. Long enough
#: for a first model load from local disk; a first-ever download can take
#: longer, and ``--timeout`` raises it.
DEFAULT_WAIT_SECONDS = 120.0
_POLL_SECONDS = 1.0

_STATE_WORDS = {
    "not_ready": "still starting",
    "warming": "loading the embedding model",
    "serving_degraded": "serving, but semantic recall is not healthy",
    "serving_full": "ready",
}


@dataclass(frozen=True)
class WarmupVerdict:
    """The outcome of waiting for a daemon to become ready."""

    ready: bool
    reachable: bool
    waited_seconds: float
    detail: str


def is_fully_warm(health: dict | None) -> bool:
    """True only when the daemon is ready AND its embedding model is warm."""
    if not isinstance(health, dict):
        return False
    return health.get("ready") is True and health.get("embedding_warm") is True


def describe_not_ready(health: dict | None) -> str:
    """One plain sentence on why a daemon is not ready yet.

    4.1.22: "warming" used to read identically whether the embedding model
    was genuinely still loading or had already exhausted its retries (a
    first run offline with no cached model never gets past that point) --
    both looked like "give it a moment." When the daemon recorded a reason
    the last attempt actually failed, that reason is included instead of
    implying progress that stopped a while ago.
    """
    if not isinstance(health, dict):
        return "the daemon stopped answering"
    state = str(health.get("runtime_state") or "")
    words = _STATE_WORDS.get(state)
    if words is None:
        engine = health.get("engine", "unknown")
        words = f"engine state '{engine}'"
    warmup_error = (health.get("readiness") or {}).get("embedding_warmup_error")
    if state == "warming":
        if warmup_error:
            return f"{words} -- last attempt failed: {warmup_error}"
        return words                      # already says the model is loading
    model = "warm" if health.get("embedding_warm") is True else "not loaded yet"
    if model == "not loaded yet" and warmup_error:
        model = f"not loaded yet (last attempt failed: {warmup_error})"
    return f"{words} (embedding model: {model})"


def wait_until_warm(
    fetch_health: Callable[[], dict | None],
    timeout_seconds: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_first_wait: Callable[[], None] | None = None,
) -> WarmupVerdict:
    """Poll /health until the daemon is fully warm or the deadline passes."""
    start = clock()
    deadline = start + max(0.0, timeout_seconds)
    health = fetch_health()
    while True:
        if is_fully_warm(health):
            return WarmupVerdict(True, True, clock() - start, "ready")
        if health is None:
            return WarmupVerdict(False, False, clock() - start,
                                 describe_not_ready(None))
        if clock() >= deadline:
            return WarmupVerdict(False, True, clock() - start,
                                 describe_not_ready(health))
        if on_first_wait is not None:
            on_first_wait()
            on_first_wait = None
        sleep(_POLL_SECONDS)
        health = fetch_health()
