"""A recall counts as in flight exactly as long as its work runs.

Background work (embedding, graph write-back, vector repair) waits while a
recall is in flight. A count that leaks upward makes every later background
step wait its full bound; a count that drops while the recall's engine call
still runs lets background work compete with it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.core import recall_gate
from superlocalmemory.server import recall_core


def _call(fast: bool = False) -> recall_core.RecallCall:
    return recall_core.RecallCall(query="q", limit=3, session_id="", agent_id="t", fast=fast)


@pytest.fixture
def patched(monkeypatch):
    """Engine work replaced by controllable stand-ins; one deep-recall slot."""
    from superlocalmemory.server import profile_runtime, recall_fallback

    snapshot = SimpleNamespace(profile_id="default", generation=1)
    monkeypatch.setattr(profile_runtime, "get_profile_runtime",
                        lambda _state: SimpleNamespace(snapshot=snapshot))
    monkeypatch.setattr(recall_fallback, "recall_keyword_fallback",
                        lambda *a, **k: {"ok": True, "fallback": True})
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: {"ok": True})
    monkeypatch.setattr(recall_core, "RECALL_SEMAPHORE", asyncio.Semaphore(1))
    base = recall_gate.in_flight()
    yield base
    assert recall_gate.in_flight() == base


def _wait_until(predicate, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_recall_cancelled_while_queued_leaves_no_count(patched, monkeypatch):
    base = patched
    monkeypatch.setattr(recall_core, "_engine_call", lambda *a: object())

    async def scenario() -> None:
        await recall_core.RECALL_SEMAPHORE.acquire()  # every deep slot taken
        task = asyncio.ensure_future(recall_core.run_recall(None, _call(), app_state=None))
        for _ in range(5):
            await asyncio.sleep(0)
        assert recall_gate.in_flight() == base + 1  # queued: counted
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        recall_core.RECALL_SEMAPHORE.release()

    asyncio.run(scenario())
    assert recall_gate.in_flight() == base


def test_the_count_holds_until_an_abandoned_engine_call_ends(patched, monkeypatch):
    base = patched
    release = threading.Event()
    started = threading.Event()

    def slow_engine_call(*_a):
        started.set()
        release.wait(30)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", slow_engine_call)
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.2")

    # A long-lived loop, as in the daemon: asyncio.run would wait for the
    # executor thread at shutdown and hide what a running server sees.
    loop = asyncio.new_event_loop()
    try:
        body = loop.run_until_complete(recall_core.run_recall(None, _call(), app_state=None))
        assert body.get("fallback") is True
        assert started.is_set()
        # The keyword answer was served, but the engine call still runs.
        assert recall_gate.in_flight() == base + 1
    finally:
        release.set()
        assert _wait_until(lambda: recall_gate.in_flight() == base)
        loop.close()


def test_an_ordinary_recall_ends_at_zero(patched, monkeypatch):
    base = patched
    monkeypatch.setattr(recall_core, "_engine_call", lambda *a: object())
    for fast in (False, True):
        body = asyncio.run(recall_core.run_recall(None, _call(fast), app_state=None))
        assert body == {"ok": True}
        assert recall_gate.in_flight() == base


def test_yield_to_recalls_returns_when_the_last_recall_ends():
    hold = recall_gate.RecallHold()
    done = threading.Event()

    def background() -> None:
        recall_gate.yield_to_recalls(30.0)
        done.set()

    thread = threading.Thread(target=background)
    thread.start()
    assert not done.wait(0.2)  # held while the recall runs
    hold.leave()
    assert done.wait(10)
    thread.join(10)
