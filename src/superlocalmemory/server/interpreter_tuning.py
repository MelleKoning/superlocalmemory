# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""How quickly the daemon hands the interpreter lock to a waiting thread.

WHY
---
A recall is mostly database reads, and the SQLite module gives up the
interpreter lock around every row it fetches. Getting it back means waiting
for whichever thread holds it to reach a switch point. With nothing else busy
that is immediate; with one CPU-bound background thread (an index build, a
graph refresh, a repair pass) each win-back waits up to the switch interval,
5 ms by default -- once per row. Measured with one busy thread beside the
reads, on a copy of a 22k-fact store:

    60 facts by id         15 ms alone ->   518 ms  (0.5 ms interval:  64 ms)
    300 visibility checks  10 ms alone -> 2,574 ms  (0.5 ms interval: 242 ms)

and the busy thread lost about 4% of its throughput at 0.5 ms. The daemon's
first minute after a start is exactly that mix: recalls beside warm-up and
repair threads.

WHAT THIS DOES
--------------
Sets the daemon's switch interval to 0.5 ms at start. ``SLM_GIL_SWITCH_INTERVAL_MS``
overrides it (a positive number of milliseconds; anything else is ignored).
One-shot commands keep Python's default.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

DEFAULT_SWITCH_INTERVAL_S = 0.0005
ENV = "SLM_GIL_SWITCH_INTERVAL_MS"


def apply_switch_interval() -> float:
    """Set and return the interval in seconds."""
    seconds = DEFAULT_SWITCH_INTERVAL_S
    raw = os.environ.get(ENV, "").strip()
    if raw:
        try:
            value = float(raw) / 1000.0
            if value > 0:
                seconds = value
        except ValueError:
            logger.warning("%s=%r is not a number of milliseconds; using %.1f ms",
                           ENV, raw, DEFAULT_SWITCH_INTERVAL_S * 1000)
    sys.setswitchinterval(seconds)
    return seconds


__all__ = ["DEFAULT_SWITCH_INTERVAL_S", "ENV", "apply_switch_interval"]
