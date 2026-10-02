# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The bounded-loop gate keeps the answer check and never pays for the reorder.

The gate refuses a match the check calls insufficient — that is the feature, so
the check stays. But each lap's recall used to run the opt-in reorder too: a
listwise question over three throwaway results, on every lap, with a different
calibration from the plain check. The gate now asks for the plain check only,
so a lap costs at most one judgement of at most three memories.
"""

from __future__ import annotations

import asyncio
import inspect

from superlocalmemory.mcp.tools_loops import register_loop_tools
from superlocalmemory.retrieval.answer_check_status import REQUEST_NO_REORDER
from tests.mcp.test_loop_tools import _Capture, _FakeEngine, _FakePool


class _RecordingEngine(_FakeEngine):
    calls: list = []

    def recall(self, query, limit=3, fast=True, **kw):
        type(self).calls.append({"limit": limit, **kw})
        return super().recall(query, limit=limit, fast=fast)


class _ForwardingPool(_FakePool):
    """A pool whose transport forwards the per-recall request."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list = []

    def recall(self, query, limit=3, fast=True, answer_check=None):
        self.requests.append(answer_check)
        return super().recall(query, limit=limit, fast=fast)


def _run_loop(cap) -> dict:
    return asyncio.run(cap.fns["slm_loop_run"](
        name="gate", gate_query="MATCH build passed", max_iterations=2,
        poll_interval_s=0.25))


def test_the_in_process_gate_asks_for_the_check_without_the_reorder() -> None:
    _RecordingEngine.calls = []
    cap = _Capture()
    register_loop_tools(cap, _RecordingEngine)
    out = _run_loop(cap)
    assert out["ok"] is True
    gate_calls = [c for c in _RecordingEngine.calls if c["limit"] == 3]
    assert gate_calls and all(c.get("answer_check") == REQUEST_NO_REORDER
                              for c in gate_calls)


def test_a_pool_that_forwards_the_request_gets_it() -> None:
    pool = _ForwardingPool()
    cap = _Capture()
    register_loop_tools(cap, _FakeEngine, get_pool=lambda: pool)
    _run_loop(cap)
    assert pool.requests and set(pool.requests) == {REQUEST_NO_REORDER}


def test_a_pool_that_cannot_forward_it_still_works() -> None:
    pool = _FakePool()  # recall(query, limit, fast) only
    cap = _Capture()
    register_loop_tools(cap, _FakeEngine, get_pool=lambda: pool)
    out = _run_loop(cap)
    assert out["ok"] is True and pool.recalled


def test_the_tool_says_what_each_lap_costs() -> None:
    cap = _Capture()
    register_loop_tools(cap, _FakeEngine)
    doc = inspect.getdoc(cap.fns["slm_loop_run"]) or ""
    assert "answer check" in doc and "each lap" in doc
    assert "billed" in doc
