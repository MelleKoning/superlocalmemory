# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What a remote reranker may cost a recall, and when to stop asking it.

WHY THIS EXISTS
    4.1.18 made a configured remote reranker run on every recall (4.1.17 never
    called it at all). Nothing then bounded what a sick endpoint could cost: a
    server that accepts the connection and never answers held each recall for
    two attempts of the 15 s read timeout -- 30.5 s measured -- and the next
    recall paid the same again, for ever. The recall ceiling is 2.0 s.

TWO GUARDS, BOTH HERE
    ``call_within``     one wall-clock deadline for the whole request, retries
                        included. The request runs on a short-lived daemon
                        thread and the recall waits for it at most the
                        deadline; a request still running then is abandoned
                        (its own transport timeouts end it soon after), so a
                        recall can never wait longer than the deadline no
                        matter how the endpoint misbehaves -- slow connect,
                        no reply, or a reply that trickles in.
    ``CircuitBreaker``  after the endpoint fails, recalls stop calling it for a
                        cooldown and get fusion order at once. When the
                        cooldown ends, ONE background probe checks it; a
                        recall never waits on that probe. A passing probe lets
                        the next real request through as a trial: success
                        closes the breaker, failure re-opens it for twice as
                        long (capped), so an endpoint that answers a one-line
                        probe but not a full batch costs one recall per
                        cooldown, not every recall.

WHAT THIS COSTS IN ANSWER QUALITY (stated, not hidden)
    * An endpoint slower than the deadline never reranks: those recalls return
      fusion order, reported as ``remote_unavailable`` and logged. Before, they
      were reranked after a wait the recall ceiling does not allow.
    * While the breaker is open (10 s, doubling to at most 5 min while the
      endpoint stays down) recalls are not reranked even if the endpoint has
      just come back; the next probe notices within one cooldown.

Pure standard library: no httpx here, so this is testable without a network.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Consecutive non-timeout failures that open the breaker. A timeout opens it
#: at once: it already cost a recall the whole deadline, and the next one would
#: pay the same.
FAILURE_THRESHOLD = 3
BASE_COOLDOWN_S = 10.0
MAX_COOLDOWN_S = 300.0

#: Requests in flight at once, abandoned ones included. A slot is held until
#: the request's own transport timeouts end it, so a hung endpoint cannot pile
#: up threads: past this many, a recall skips the endpoint instead. Sized well
#: above any realistic number of recalls at once, so healthy concurrent recalls
#: never meet it; a first draft at 4 cost 8 of 12 concurrent recalls their
#: reranking against a healthy endpoint (test_remote_reranker_breaker).
MAX_IN_FLIGHT = 64


class DeadlineExceeded(Exception):
    """The request did not finish inside its deadline; it was abandoned."""


class TooManyInFlight(Exception):
    """Every request slot is held by a request that has not finished yet."""


def call_within(
    fn: Callable[[float], T],
    deadline_s: float,
    slots: threading.BoundedSemaphore,
    *,
    name: str = "remote-rerank",
) -> T:
    """Run ``fn(deadline_at)`` and wait for it at most ``deadline_s`` seconds.

    ``fn`` receives the absolute ``time.monotonic()`` deadline so it can stop
    retrying once the budget is spent. It runs on a daemon thread that owns one
    of ``slots`` until it returns, whether or not anyone is still waiting.

    Raises:
        TooManyInFlight: no slot was free; ``fn`` never ran.
        DeadlineExceeded: ``fn`` was still running at the deadline.
        Exception: whatever ``fn`` raised, re-raised in the caller's thread.
    """
    if not slots.acquire(blocking=False):
        raise TooManyInFlight(f"{name}: all request slots are busy")
    deadline_at = time.monotonic() + max(0.0, deadline_s)
    box: dict[str, object] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            box["value"] = fn(deadline_at)
        except BaseException as exc:  # noqa: BLE001 -- handed to the caller
            box["error"] = exc
        finally:
            slots.release()
            done.set()

    worker = threading.Thread(target=_run, name=name, daemon=True)
    try:
        worker.start()
    except BaseException:
        slots.release()
        raise
    if not done.wait(timeout=max(0.0, deadline_at - time.monotonic())):
        raise DeadlineExceeded(
            f"{name}: no answer within {deadline_s:g}s; request abandoned",
        )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]


@dataclass(frozen=True)
class BreakerDecision:
    """What a caller may do right now."""

    call: bool          # send the real request
    start_probe: bool   # the caller should launch the (single) background probe


class CircuitBreaker:
    """Closed -> open -> (background probe) -> trial -> closed. Thread-safe."""

    CLOSED, OPEN, TRIAL = "closed", "open", "trial"

    def __init__(
        self,
        *,
        failure_threshold: int = FAILURE_THRESHOLD,
        base_cooldown_s: float = BASE_COOLDOWN_S,
        max_cooldown_s: float = MAX_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = max(1, int(failure_threshold))
        self._base = max(0.0, float(base_cooldown_s))
        self._max = max(self._base, float(max_cooldown_s))
        self._clock = clock
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures = 0
        self._cooldown = self._base
        self._open_until = 0.0
        self._probing = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def cooldown_s(self) -> float:
        with self._lock:
            return self._cooldown

    def decide(self) -> BreakerDecision:
        """Whether to call now, and whether this caller should start the probe."""
        with self._lock:
            if self._state != self.OPEN:
                return BreakerDecision(call=True, start_probe=False)
            if self._probing or self._clock() < self._open_until:
                return BreakerDecision(call=False, start_probe=False)
            self._probing = True
            return BreakerDecision(call=False, start_probe=True)

    def record_success(self) -> bool:
        """A real request succeeded. True when this closed an open/trial breaker."""
        with self._lock:
            reopened = self._state != self.CLOSED
            self._state = self.CLOSED
            self._failures = 0
            self._cooldown = self._base
            return reopened

    def record_failure(self, *, timed_out: bool) -> bool:
        """A real request failed. True when this opened the breaker."""
        with self._lock:
            if self._state == self.OPEN:
                return False
            self._failures += 1
            if self._state == self.TRIAL:
                # The probe passed but real work still fails: back off harder.
                self._cooldown = min(self._max, max(self._base, self._cooldown * 2))
                self._open(self._cooldown)
                return True
            if timed_out or self._failures >= self._threshold:
                self._open(self._cooldown)
                return True
            return False

    def probe_finished(self, ok: bool) -> None:
        """The background probe ended; ``ok`` lets one trial request through."""
        with self._lock:
            self._probing = False
            if self._state != self.OPEN:
                return
            if ok:
                self._state = self.TRIAL
                return
            self._cooldown = min(self._max, max(self._base, self._cooldown * 2))
            self._open(self._cooldown)

    def _open(self, seconds: float) -> None:
        self._state = self.OPEN
        self._open_until = self._clock() + seconds
