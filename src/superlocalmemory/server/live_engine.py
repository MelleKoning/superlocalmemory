# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which engine the daemon closes when it stops.

Until 4.1.22 the shutdown closed the engine built at start-up. A settings
change from the dashboard, or a model switch, publishes a new engine and closes
the old one, so at stop the start-up engine was already closed and the LIVE
one, with its embedding and reranker workers, was never closed: those workers
lived on until their parent watchdog noticed the daemon had gone.
"""

from __future__ import annotations

from typing import Any


def engine_to_close(app_state: Any, started: Any) -> Any:
    """The engine currently published, falling back to the one start-up built."""
    return getattr(app_state, "engine", None) or started


__all__ = ["engine_to_close"]
