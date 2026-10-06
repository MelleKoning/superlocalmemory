# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Load-robust timing assertions for hook hot-path budget tests (4.1.22).

A plain wall-clock tail assertion (``p95 duration < N ms`` over a few dozen-
to-hundred samples) is not robust to a shared/loaded host (see LANE-RULES:
several lanes run pytest on this machine concurrently). That was W4
(``test_post_tool_hook_under_10ms_p95`` measured p95 49-197 ms vs. a 30 ms
ceiling under sibling-test load, passing alone): sibling-lane contention on a
shared host causes an occasional, sporadic scheduling delay for THIS
process -- not a sustained slowdown -- and p95 over a modest sample size is
exactly the statistic a minority of unlucky delays contaminates most, because
it looks directly at the tail.

An interleaved same-run "no-op" calibration was tried first and rejected: a
near-instant no-op has almost no chance of landing on a scheduling-quantum
boundary, so its own p95 stays near zero even while the real (longer,
multi-step) hot path gets hit -- the two do not scale together, so
subtracting one from the other does not cancel the noise it was meant to.

The median does, because a scheduling delay hits a MINORITY of calls, not
the typical one: the median call is normally not among the delayed ones, so
it reports true steady-state cost even on a loaded host. A genuine
regression -- CPU-bound or a blocking wait, including a plain
``time.sleep`` -- instead adds its cost to EVERY call, which shifts the
whole distribution including the median, so the check stays sensitive to
real slowdowns. This is the same technique already proven against this exact
failure mode elsewhere in this suite
(``tests/test_hooks/test_hook_handlers.py::test_user_prompt_under_budget``).
"""

from __future__ import annotations

import statistics
import time
from typing import Callable


def assert_median_under_budget(
    measure_once: Callable[[], None],
    *,
    budget_ms: float,
    iterations: int = 100,
    label: str = "hot path",
) -> None:
    """Assert the median wall-clock cost of ``measure_once`` stays bounded.

    Robust to a minority of host-contention-delayed calls (the median
    ignores them) while staying sensitive to a regression that adds real
    cost to every call (the median shifts with the whole distribution).
    """
    durations_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        measure_once()
        durations_ms.append((time.perf_counter_ns() - t0) / 1e6)

    median_ms = statistics.median(durations_ms)
    p95_ms = sorted(durations_ms)[int(len(durations_ms) * 0.95)]
    assert median_ms < budget_ms, (
        f"{label}: median {median_ms:.2f}ms over {iterations} calls "
        f"exceeds {budget_ms}ms budget (p95={p95_ms:.2f}ms)"
    )
