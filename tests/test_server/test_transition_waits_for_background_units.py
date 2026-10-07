# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A settings change or profile switch is not failed by a slow background unit.

A transition drains every lease within ``_DRAIN_TIMEOUT_SECS`` and holds new
requests while it does. One background unit (a memory being materialized on a
large store) could hold its lease longer than that, so the change failed with
503 though nothing was wrong (measured on a 21,847-fact store). Now new
background units are held back first and the running ones are given time to
finish, while requests are still served; only then does the drain start.
"""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.server import profile_runtime as prt


@pytest.fixture()
def short(monkeypatch):
    monkeypatch.setattr(prt, "_DRAIN_TIMEOUT_SECS", 0.3)
    monkeypatch.setattr(prt, "_BACKGROUND_SETTLE_SECS", 10.0, raising=False)


def _background_unit(runtime, started: threading.Event, release: threading.Event) -> None:
    with runtime.operation_nowait() as snap:
        assert snap is not None
        started.set()
        release.wait(10)


@pytest.mark.parametrize("kind", ["reconfigure", "switch"])
def test_a_background_unit_longer_than_the_drain_does_not_fail_the_change(short, kind) -> None:
    runtime = prt.ProfileRuntime("default")
    started, release = threading.Event(), threading.Event()
    unit = threading.Thread(target=_background_unit, args=(runtime, started, release))
    unit.start()
    assert started.wait(5)
    # The unit finishes after the drain timeout (0.3 s) has long passed.
    threading.Timer(1.0, release.set).start()
    if kind == "reconfigure":
        snapshot = runtime.reconfigure(lambda _s: None)
        assert snapshot.profile_id == "default"
    else:
        snapshot = runtime.transition("work", lambda _p, _t: None)
        assert snapshot.profile_id == "work"
    unit.join(5)


def test_requests_are_served_while_it_waits_and_new_background_units_are_held(short) -> None:
    runtime = prt.ProfileRuntime("default")
    started, release = threading.Event(), threading.Event()
    unit = threading.Thread(target=_background_unit, args=(runtime, started, release))
    unit.start()
    assert started.wait(5)
    change = threading.Thread(target=runtime.reconfigure, args=(lambda _s: None,))
    change.start()
    time.sleep(0.5)  # the change is waiting for the unit, past the drain timeout
    served = threading.Event()

    def request() -> None:
        with runtime.operation():
            served.set()

    threading.Thread(target=request).start()
    assert served.wait(2), "a request was held while the change waited for background work"
    with runtime.operation_nowait() as snap:
        assert snap is None  # no new background unit starts meanwhile
    release.set()
    change.join(5)
    assert not change.is_alive() and not runtime.transitioning
    unit.join(5)
