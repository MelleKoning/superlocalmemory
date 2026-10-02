# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recalls that are not questions skip the answer check.

Loading context — at session start, for an auto-injected resource, for a
per-prompt hook, for the daemon's own warm-up — runs a recall to fetch
memories, not to answer a question. Judging it would ask whether memories
answer a "question" nobody asked, could abstain on it, and, with the online
check on, would send those memories off the machine and bill the user's key.

Results are unaffected: a skipped check only means the recall reports no
verdict, exactly as with the check off.

``skip_answer_check()`` marks the current context. It is a ``ContextVar``, so
``asyncio.to_thread`` and ``contextvars.copy_context().run`` carry it; a plain
thread or another process does not — callers on the far side of HTTP send
``ANSWER_CHECK_PARAM=ANSWER_CHECK_SKIP`` and the route re-enters the marker.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

#: Query parameter (and value) a caller sends so the daemon skips the check.
ANSWER_CHECK_PARAM = "answer_check"
ANSWER_CHECK_SKIP = "skip"

_SKIPPED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "slm_answer_check_skipped", default=False,
)


@contextlib.contextmanager
def skip_answer_check() -> Iterator[None]:
    """Within this block, recalls report no verdict and send nothing to be judged."""
    token = _SKIPPED.set(True)
    try:
        yield
    finally:
        _SKIPPED.reset(token)


def answer_check_skipped() -> bool:
    """Whether the current context asked for the answer check to be skipped."""
    return _SKIPPED.get()


def wants_skip(value: object) -> bool:
    """Whether a wire value (the query parameter) asks for the skip."""
    return isinstance(value, str) and value.strip().lower() == ANSWER_CHECK_SKIP


__all__ = [
    "ANSWER_CHECK_PARAM",
    "ANSWER_CHECK_SKIP",
    "answer_check_skipped",
    "skip_answer_check",
    "wants_skip",
]
