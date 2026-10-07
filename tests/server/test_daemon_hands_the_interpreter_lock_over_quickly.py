# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon hands the interpreter lock to a waiting recall within 0.5 ms, not 5 ms.

A recall's database reads give up the interpreter lock for every row fetched
and must win it back. With one busy background thread (an index build, a
graph refresh) each win-back waited up to Python's 5 ms switch interval.
Measured: 60 facts by id 15 -> 518 ms, 300 visibility checks 10 -> 2,574 ms;
with a 0.5 ms interval 64 ms and 242 ms, the busy thread losing ~4% throughput.
"""

from __future__ import annotations

import inspect
import sys

import pytest

from superlocalmemory.server import interpreter_tuning


@pytest.fixture(autouse=True)
def _restore_interval():
    before = sys.getswitchinterval()
    yield
    sys.setswitchinterval(before)


def test_the_default_is_half_a_millisecond(monkeypatch) -> None:
    monkeypatch.delenv("SLM_GIL_SWITCH_INTERVAL_MS", raising=False)
    sys.setswitchinterval(0.005)
    assert interpreter_tuning.apply_switch_interval() == pytest.approx(0.0005)
    assert sys.getswitchinterval() == pytest.approx(0.0005)


def test_an_operator_can_set_it_and_a_bad_value_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("SLM_GIL_SWITCH_INTERVAL_MS", "2")
    assert interpreter_tuning.apply_switch_interval() == pytest.approx(0.002)
    monkeypatch.setenv("SLM_GIL_SWITCH_INTERVAL_MS", "nonsense")
    assert interpreter_tuning.apply_switch_interval() == pytest.approx(0.0005)
    monkeypatch.setenv("SLM_GIL_SWITCH_INTERVAL_MS", "0")
    assert interpreter_tuning.apply_switch_interval() == pytest.approx(0.0005)


def test_the_daemon_applies_it_at_start() -> None:
    from superlocalmemory.server import unified_daemon

    assert "apply_switch_interval()" in inspect.getsource(unified_daemon.start_server)
