# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A save is not refused because this process stalled while saving it.

Reproduces the boundary-pack failure deterministically: one save in flight,
the journal idle, and the whole process frozen for longer than the save's
2 s journal budget right after the journal writer picked the save up (one
thread holding the interpreter, as a loaded or memory-starved machine does).
The caller's clock ran out during the freeze, so the save used to be
withdrawn and refused with "too many saves are arriving at once", though
nothing else was saving and the journal was milliseconds from durable.
"""

from __future__ import annotations

import random
import threading
import time

from tests.core.test_remember_contention_accept import (  # noqa: F401 - fixtures
    _actor,
    _HeldWriteLock,
    _request,
    data_dir,
    runtime,
)

_JOURNAL_BUDGET_S = 2.0
_STALL_S = 2.6


def _interpreter_hog(seconds: float) -> list[float]:
    """A list whose sort holds the interpreter lock for about *seconds*."""
    sample = [random.random() for _ in range(500_000)]
    started = time.monotonic()
    sample.sort()
    per_item = max(1e-9, (time.monotonic() - started) / len(sample))
    return [random.random() for _ in range(int(seconds / per_item))]


def test_a_save_is_accepted_when_the_process_stalls_mid_save(runtime, data_dir):  # noqa: F811
    hog = _interpreter_hog(_STALL_S)
    writer = runtime.journal._writer
    armed, fire = threading.Event(), threading.Event()
    stall = threading.Thread(target=lambda: (fire.wait(), hog.sort()), daemon=True)
    stall.start()
    real_claim = writer._claim

    def claim_then_stall(ops):
        live = real_claim(ops)
        if armed.is_set() and not fire.is_set():
            fire.set()
            time.sleep(0.01)  # hand the interpreter to the stalling thread
        return live

    writer._claim = claim_then_stall
    lock = _HeldWriteLock(data_dir / "memory.db")  # memory.db busy: answer is "accepted"
    try:
        armed.set()
        started = time.monotonic()
        receipt = runtime.remember(
            _request("stalled-1"), _actor(),
            deadline_ms=int(_JOURNAL_BUDGET_S * 1000), accept_after_ms=1_200,
        )
        elapsed = time.monotonic() - started
    finally:
        lock.release()
        stall.join(timeout=30)
    assert fire.is_set(), "the stall never fired; the test proved nothing"
    assert elapsed > _JOURNAL_BUDGET_S, f"no stall past the budget ({elapsed:.2f}s)"
    assert receipt.payload["status"] == "accepted"
    assert receipt.payload["durable"] is True
    assert runtime.wait_for_deferred(timeout=15.0)
    assert runtime.journal.get(receipt.payload["admission_id"]).state == "committed"
