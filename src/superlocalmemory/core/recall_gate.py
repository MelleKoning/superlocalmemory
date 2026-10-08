# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""v3.4.32: Recall-in-flight counter used to give /search priority over the
pending materializer.

Every recall handler calls ``begin_recall()`` on entry and ``end_recall()``
in a finally block. The pending-memory materializer thread polls
``in_flight()`` and sleeps while any recall is active, so the shared
embedder worker never serves a materialization ahead of a user-initiated
recall.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

_condition = threading.Condition(threading.Lock())
_active = 0
_work_context = threading.local()


def begin_recall() -> None:
    global _active
    with _condition:
        _active += 1


def end_recall() -> None:
    global _active
    with _condition:
        _active = max(0, _active - 1)
        if _active == 0:
            _condition.notify_all()


def in_flight() -> int:
    with _condition:
        return _active


class RecallHold:
    """One recall's place in the in-flight count, held until its work settles.

    The recall's coroutine holds it, and so does each piece of engine work it
    hands to a thread. The count drops exactly once, when the last holder lets
    go: a recall answered by the keyword fallback while its engine call still
    runs keeps counting until that call ends, and a recall cancelled before its
    work started drops at once. Work that has not started when the coroutine
    leaves is skipped, because nobody is left to read its answer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holders = 1
        self._owner_gone = False
        begin_recall()

    def run(self, fn: Callable[..., object], *args: object) -> object:
        """Run ``fn`` on the calling thread while holding the recall in flight."""
        with self._lock:
            if self._owner_gone:
                return None  # the recall already answered or was cancelled
            self._holders += 1
        try:
            return fn(*args)
        finally:
            self._release()

    def leave(self) -> None:
        """The recall's coroutine is done; work still running keeps the count."""
        with self._lock:
            if self._owner_gone:
                return
            self._owner_gone = True
        self._release()

    def _release(self) -> None:
        with self._lock:
            self._holders -= 1
            last = self._holders == 0
        if last:
            end_recall()


def yield_to_recalls(max_seconds: float = 30.0) -> None:
    """Wait while any recall is in flight, at most ``max_seconds``.

    For background writers that share the store with recall: steady recall
    traffic slows them, but can never stop them.
    """
    deadline = time.monotonic() + max(0.0, max_seconds)
    with _condition:
        while _active > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            _condition.wait(timeout=min(remaining, 0.1))


@contextmanager
def background_work(
    *,
    preempt_requested: Callable[[], bool] | None = None,
) -> Iterator[None]:
    """Mark best-effort work that must yield shared inference to recall.

    The marker is thread-local because materialization, health probes, and
    interactive handlers all share one resident engine and one embedder.
    Nested callers restore the previous marker on exit.  A daemon-owned
    materializer may also provide a preemption callback for a profile/runtime
    reconfigure.  Inference clients use that callback to cut a bounded
    background request short instead of holding the transition drain lease.
    """
    previous = bool(getattr(_work_context, "background", False))
    previous_preempt = getattr(_work_context, "preempt_requested", None)
    _work_context.background = True
    _work_context.preempt_requested = (
        preempt_requested if preempt_requested is not None else previous_preempt
    )
    try:
        yield
    finally:
        _work_context.background = previous
        _work_context.preempt_requested = previous_preempt


def is_background_work() -> bool:
    """Return whether the current thread is running best-effort work."""
    return bool(getattr(_work_context, "background", False))


def background_preempt_requested() -> bool:
    """Return whether daemon-owned background work must release its lease."""
    callback = getattr(_work_context, "preempt_requested", None)
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception:
        # A status read is advisory.  It must not crash a materializer or turn
        # a valid recall into an ingestion failure.
        return False


@contextmanager
def idle_wait_deadline(deadline: float) -> Iterator[None]:
    """Bound ``wait_for_foreground_idle`` on this thread to ``deadline``.

    ``deadline`` is a ``time.monotonic()`` value. Opt-in: callers that never
    enter this context wait exactly as before. A job that may yield for a
    long time under steady recall load (memory-kind backfill) uses it so its
    thread is never stuck inside a wait it cannot leave; it gives that batch
    up and tries again later instead.
    """
    previous = getattr(_work_context, "idle_deadline", None)
    _work_context.idle_deadline = deadline
    try:
        yield
    finally:
        _work_context.idle_deadline = previous


def wait_for_foreground_idle() -> None:
    """Block background inference while an interactive recall is active."""
    if not is_background_work():
        return
    deadline = getattr(_work_context, "idle_deadline", None)
    with _condition:
        while _active > 0:
            if deadline is not None and time.monotonic() >= deadline:
                return
            _condition.wait(timeout=0.1)
