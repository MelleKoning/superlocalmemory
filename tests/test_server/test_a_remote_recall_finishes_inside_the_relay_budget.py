# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A recall relayed through the gateway answers inside the relay's time.

The gateway gives a relayed call 25 s and a recall's own budget is 25 s, so a
slow recall's keyword answer was ready exactly when the relay gave up. A call
may now carry a shorter budget (``budget_s``): it can only SHORTEN the budget,
never extend it, and a call without one behaves exactly as before.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.server import recall_core
from tests.test_server.test_canonical_remember_route import _client


def _call(**extra) -> recall_core.RecallCall:
    return recall_core.RecallCall(
        query="q", limit=3, session_id="", agent_id="t", fast=True, **extra)


@pytest.fixture
def slow_engine(monkeypatch):
    """An engine call that stays busy until released; the keyword answer is a marker."""
    from superlocalmemory.server import profile_runtime, recall_fallback

    release = threading.Event()
    snapshot = SimpleNamespace(profile_id="default", generation=1)
    monkeypatch.setattr(profile_runtime, "get_profile_runtime",
                        lambda _state: SimpleNamespace(snapshot=snapshot))
    monkeypatch.setattr(recall_fallback, "recall_keyword_fallback",
                        lambda *a, **k: {"fallback": True})
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: {"fallback": False})
    monkeypatch.setattr(recall_core, "_engine_call",
                        lambda *a: release.wait(10) and object())
    monkeypatch.delenv("SLM_SEARCH_RECALL_TIMEOUT_S", raising=False)
    yield release
    release.set()


def _serve(call: recall_core.RecallCall) -> tuple[dict, float]:
    loop = asyncio.new_event_loop()
    try:
        started = time.monotonic()
        body = loop.run_until_complete(recall_core.run_recall(None, call, app_state=None))
        return body, time.monotonic() - started
    finally:
        loop.close()


def test_a_call_budget_ends_the_wait_early_with_the_keyword_answer(slow_engine, monkeypatch):
    monkeypatch.setattr(recall_core, "RECALL_BUDGET_FLOOR_S", 0.05)
    body, took = _serve(_call(budget_s=0.2))
    assert body == {"fallback": True}
    assert 0.15 <= took < 2.0, took  # the default budget is 25 s


def test_a_call_budget_never_extends_the_default_budget(slow_engine, monkeypatch):
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.3")
    body, took = _serve(_call(budget_s=10.0))
    assert body == {"fallback": True}
    assert 0.25 <= took < 2.0, took


def test_without_a_call_budget_the_default_budget_applies_unchanged(slow_engine, monkeypatch):
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.4")
    body, took = _serve(_call())
    assert body == {"fallback": True}
    assert 0.35 <= took < 2.0, took


def test_a_fast_recall_inside_its_call_budget_is_the_ordinary_answer(slow_engine, monkeypatch):
    slow_engine.set()
    body, _ = _serve(_call(budget_s=5.0))
    assert body == {"fallback": False}


def test_the_effective_budget_is_the_shorter_one_with_a_floor():
    assert recall_core.RECALL_BUDGET_FLOOR_S == 2.0
    assert recall_core.effective_budget_s(_call()) == recall_core.recall_budget_s()
    assert recall_core.effective_budget_s(_call(budget_s=9.5)) == 9.5
    assert recall_core.effective_budget_s(_call(budget_s=500.0)) == recall_core.recall_budget_s()
    # A tiny remaining time still leaves the fallback query time to run.
    assert recall_core.effective_budget_s(_call(budget_s=0.01)) == 2.0
    assert recall_core.effective_budget_s(_call(budget_s=-4.0)) == 2.0


def test_the_floor_never_raises_a_budget_above_the_default(monkeypatch):
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.5")
    assert recall_core.effective_budget_s(_call(budget_s=0.01)) == 0.5


@pytest.mark.parametrize("raw, expected", [
    ("12.5", 12.5), ("3", 3.0), ("0.25", 0.25),
    ("", None), ("abc", None), ("nan", None), ("inf", None), ("-inf", None),
    ("0", None), ("-3", None), ("1e999", None), (" ", None),
])
def test_the_route_reads_a_budget_and_ignores_anything_invalid(
    engine_with_mock_deps, monkeypatch, raw, expected,
):
    seen: list = []

    async def fake_run_recall(engine, call, *, app_state):
        seen.append(call.budget_s)
        return {"results": [], "count": 0}

    monkeypatch.setattr(recall_core, "run_recall", fake_run_recall)
    params = {"q": "anything"}
    if raw != "":
        params["budget_s"] = raw
    with _client(engine_with_mock_deps) as client:
        response = client.get("/recall", params=params)
    assert response.status_code == 200, response.text
    assert seen == [expected]
