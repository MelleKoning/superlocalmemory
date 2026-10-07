# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The 1.5 s remember ceiling counts from the request, not from the durable write.

The inline enrichment wait was sized from a clock started AFTER the trust and
policy checks, plus 0.25 s of slack on top. Measured from a separate client on
a copy of a 22k-fact store while the daemon warmed up: remembers of 0.9-1.6 s,
the time spent waiting for the busy embedding worker inside that over-sized
budget. The memory is saved either way; only the wait for its vector is
bounded, and the background pass attaches the vector later.
"""

from __future__ import annotations

import asyncio
import time as real_time
from types import SimpleNamespace

from tests.test_server.test_canonical_remember_route import _client


class _Clock:
    """``time`` with a monotonic clock the test can move forward."""

    def __init__(self) -> None:
        self.offset = 0.0

    def __getattr__(self, name):
        return getattr(real_time, name)

    def monotonic(self) -> float:
        return real_time.monotonic() + self.offset


def test_slow_checks_before_the_write_shrink_the_enrichment_wait(
    engine_with_mock_deps, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon as daemon

    clock = _Clock()
    monkeypatch.setattr(daemon, "time", clock)
    seen: dict[str, float] = {}
    monkeypatch.setattr(daemon, "_enrich_and_release",
                        lambda engine, fact_ids, budget, **kw: (
                            seen.__setitem__("budget", budget), daemon._enrichment_semaphore.release(),
                            len(fact_ids))[2])
    real_wait_for = asyncio.wait_for

    async def spy_wait_for(aw, timeout):
        seen["timeout"] = timeout
        return await real_wait_for(aw, timeout)

    monkeypatch.setattr(daemon.asyncio, "wait_for", spy_wait_for)
    # The trust/policy checks take 0.6 s of the ceiling (a busy store).
    engine_with_mock_deps._hooks.register_pre(
        "store", lambda ctx: setattr(clock, "offset", clock.offset + 0.6))
    with _client(engine_with_mock_deps) as client:
        r = client.post("/remember", json={"content": "Asha moves the standup to 9:30 on Mondays.",
                                           "idempotency_key": "ceiling-1"})
    assert r.status_code == 200, r.text
    assert "timeout" in seen, "inline enrichment did not run"
    remaining = daemon._REMEMBER_TOTAL_CEILING_SECONDS - 0.6
    assert seen["timeout"] <= remaining + 1e-9, (
        f"waits up to {seen['timeout']:.2f}s for the vector with only {remaining:.2f}s "
        "of the 1.5 s ceiling left")
    assert seen["budget"] <= seen["timeout"]


def test_no_time_left_means_no_inline_wait(engine_with_mock_deps, monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon as daemon

    clock = _Clock()
    monkeypatch.setattr(daemon, "time", clock)
    called: list[float] = []
    monkeypatch.setattr(daemon, "_enrich_and_release",
                        lambda engine, fact_ids, budget, **kw: (
                            called.append(budget), daemon._enrichment_semaphore.release(), 0)[2])
    engine_with_mock_deps._hooks.register_pre(
        "store", lambda ctx: setattr(clock, "offset", clock.offset + 1.4))
    with _client(engine_with_mock_deps) as client:
        r = client.post("/remember", json={"content": "Ravi ships the hotfix on Friday.",
                                           "idempotency_key": "ceiling-2"})
    assert r.status_code == 200, r.text
    assert called == [], "waited for a vector with no time left inside the ceiling"
    assert r.json()["searchable_by"] == "wording"
