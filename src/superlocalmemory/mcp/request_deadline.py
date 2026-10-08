# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""How long the current MCP request may still take, when its sender said so.

A call relayed through the gateway is abandoned by the relay at a fixed time.
The laptop that receives it stamps ``x-slm-deadline-ms`` on the request it
hands to the daemon: epoch milliseconds on THIS computer's clock, already
reduced by a margin for the keyword fallback and the trip back. A tool that can
shorten its own work (``recall``) reads what is left here.

Like the agent id (:mod:`mcp.agent_context`) this is a ContextVar, so
concurrent requests never see each other's value. A caller can only make its
own request shorter - the value feeds a budget that is capped by the default.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager

DEADLINE_HEADER = "x-slm-deadline-ms"

#: Epoch milliseconds are 13 digits today; anything longer is not a clock reading.
_MAX_DIGITS = 16

_deadline_ms: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "slm_request_deadline_ms", default=None,
)


def parse_deadline_header(value: str | bytes | None) -> int | None:
    """The integer in a header value, or ``None`` when it is not a plain one."""
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        return None
    if len(value) > _MAX_DIGITS:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def current_deadline_ms() -> int | None:
    """The current request's deadline (epoch ms), or ``None`` when it has none."""
    return _deadline_ms.get()


def remaining_budget_s(now_ms: float | None = None) -> float | None:
    """Seconds left until the request's deadline; ``None`` when it has none.

    Never zero or negative: a deadline already past yields one millisecond, so
    the recall layer's floor decides what to do rather than an ignored value.
    """
    deadline = _deadline_ms.get()
    if deadline is None:
        return None
    now = time.time() * 1000 if now_ms is None else now_ms
    return max((deadline - now) / 1000.0, 0.001)


@contextmanager
def request_deadline(deadline_ms: int | None) -> Iterator[None]:
    """Run everything inside under a request deadline (``None`` = none)."""
    token = _deadline_ms.set(deadline_ms)
    try:
        yield
    finally:
        _deadline_ms.reset(token)


__all__ = ["DEADLINE_HEADER", "current_deadline_ms", "parse_deadline_header",
           "remaining_budget_s", "request_deadline"]
