# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What the Answer Check history costs a recall: one record, microseconds.

Gates are ~10x above what was measured on Apple Silicon (median ~2 us), so
they catch an O(n) or I/O regression, not machine noise.
"""

from __future__ import annotations

import statistics
import time

from superlocalmemory.core import answer_check_history as h

from ..test_core.test_answer_check_history import make_response

CALLS = 50_000


def test_record_overhead_micro() -> None:
    h._reset_for_testing()
    h.enable(True)
    try:
        resp = make_response()
        for _ in range(2_000):                       # warm
            h.record_recall_verdict(resp, profile_id="default")
        samples = []
        for _ in range(CALLS):
            t0 = time.perf_counter_ns()
            h.record_recall_verdict(resp, profile_id="default")
            samples.append(time.perf_counter_ns() - t0)
        samples.sort()
        median_us = statistics.median(samples) / 1000
        p99_us = samples[int(len(samples) * 0.99)] / 1000
        print(f"\nrecord_recall_verdict: median {median_us:.2f} us, p99 {p99_us:.2f} us "
              f"over {CALLS} calls (ring full: {len(h._ring) == h.RING_CAPACITY})")
        assert median_us < 20.0, median_us
        assert p99_us < 100.0, p99_us
    finally:
        h._reset_for_testing()
