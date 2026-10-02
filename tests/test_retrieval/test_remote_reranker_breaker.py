# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A sick remote reranker costs a recall one bounded wait, then nothing.

Reproduces audit H-2: an endpoint that accepts the connection and never
answers held every recall for 2 attempts x 15 s (30.5 s measured), for ever.

Real sockets on 127.0.0.1 (servers started by the test) for the timing cases;
``httpx.MockTransport`` for the cases that only count requests. No test touches
the network beyond loopback.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from unittest.mock import patch

import httpx
import pytest

from superlocalmemory.retrieval import remote_rerank_guard as guard
from superlocalmemory.retrieval.remote_reranker import RemoteReranker
from superlocalmemory.storage.models import AtomicFact

RECALL_CEILING_S = 2.0


def _candidates() -> list[tuple[AtomicFact, float]]:
    return [
        (AtomicFact(fact_id="f0", memory_id="m", content="zero"), 0.9),
        (AtomicFact(fact_id="f1", memory_id="m", content="one"), 0.5),
        (AtomicFact(fact_id="f2", memory_id="m", content="two"), 0.1),
    ]


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@contextmanager
def _silent_server(mode: str = "hang") -> Iterator[tuple[str, list[int]]]:
    """A loopback server. ``hang``: accept, read, never reply. ``drip``:
    send headers, then one body byte every 0.2 s."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    srv.settimeout(0.1)
    port = srv.getsockname()[1]
    accepted: list[int] = []
    stop = threading.Event()
    conns: list[socket.socket] = []

    def _serve_one(conn: socket.socket) -> None:
        try:
            conn.recv(65536)
            if mode == "drip":
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 4096\r\n\r\n"
                )
                while not stop.is_set():
                    conn.sendall(b" ")
                    time.sleep(0.2)
            else:
                stop.wait(30)
        except OSError:
            pass

    def _loop() -> None:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except (socket.timeout, OSError):
                continue
            accepted.append(1)
            conns.append(conn)
            threading.Thread(target=_serve_one, args=(conn,), daemon=True).start()

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/rerank", accepted
    finally:
        stop.set()
        for c in conns:
            try:
                c.close()
            except OSError:
                pass
        srv.close()
        t.join(timeout=2)


@contextmanager
def _mock_http(
    handler: Callable[[httpx.Request], httpx.Response],
    reranker: RemoteReranker | None = None,
):
    """Fake the socket only; ``reranker`` drops a client pooled by an earlier stub."""
    if reranker is not None:
        reranker.unload()
    requests: list[httpx.Request] = []
    real_client = httpx.Client

    def _handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def _factory(**kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(_handler), **kwargs)

    with patch("httpx.Client", _factory):
        yield requests


def _ok(request: httpx.Request) -> httpx.Response:
    n = len(json.loads(request.content)["documents"])
    return httpx.Response(200, json={"results": [
        {"index": i, "relevance_score": float(i)} for i in range(n)
    ]})


def _down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# H-2: the measured failure, reproduced on a real socket
# ---------------------------------------------------------------------------

class TestAHungEndpointIsBounded:

    def test_a_hung_endpoint_costs_one_deadline_not_thirty_seconds(self) -> None:
        with _silent_server("hang") as (url, _accepted):
            rr = RemoteReranker("m", url, deadline_seconds=0.5)
            t0 = time.monotonic()
            results, applied, status = rr.rerank_with_status("q", _candidates())
            wall = time.monotonic() - t0
            rr.shutdown()
        assert applied is False
        assert status == "remote_unavailable"
        assert [f.fact_id for f, _ in results] == ["f0", "f1", "f2"]
        assert wall < 0.5 + 0.4, f"recall waited {wall:.2f}s on a hung endpoint"

    def test_a_slow_drip_reply_cannot_stretch_the_deadline(self) -> None:
        with _silent_server("drip") as (url, _accepted):
            rr = RemoteReranker("m", url, deadline_seconds=0.5)
            t0 = time.monotonic()
            _, applied, _ = rr.rerank_with_status("q", _candidates())
            wall = time.monotonic() - t0
            rr.shutdown()
        assert applied is False
        assert wall < 0.5 + 0.4, f"a trickling reply held the recall {wall:.2f}s"

    def test_a_read_timeout_is_not_retried(self) -> None:
        with _silent_server("hang") as (url, accepted):
            rr = RemoteReranker("m", url, deadline_seconds=0.4)
            rr.rerank_with_status("q", _candidates())
            time.sleep(0.6)
            rr.shutdown()
        assert len(accepted) == 1, f"{len(accepted)} connections for one recall"

    def test_after_a_timeout_the_next_recalls_do_not_wait(self) -> None:
        with _silent_server("hang") as (url, accepted):
            rr = RemoteReranker("m", url, deadline_seconds=0.4)
            rr.rerank_with_status("q", _candidates())
            walls = []
            for _ in range(5):
                t0 = time.monotonic()
                _, applied, status = rr.rerank_with_status("q", _candidates())
                walls.append(time.monotonic() - t0)
                assert applied is False and status == "remote_unavailable"
            rr.shutdown()
        assert max(walls) < 0.05, f"later recalls still waited: {walls}"
        assert len(accepted) == 1, "the endpoint was called again while paused"

    def test_the_default_deadline_fits_inside_the_recall_ceiling(self) -> None:
        # Stock config passes cross_encoder_timeout_seconds=15.0.
        rr = RemoteReranker("m", "http://127.0.0.1:9/v1", timeout_seconds=15.0)
        assert 0 < rr.deadline_seconds < RECALL_CEILING_S

    def test_a_tighter_configured_timeout_still_wins(self) -> None:
        rr = RemoteReranker("m", "http://127.0.0.1:9/v1", timeout_seconds=0.3)
        assert rr.deadline_seconds == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# The breaker: stop calling, probe in the background, trial, close
# ---------------------------------------------------------------------------

class TestTheBreaker:

    def _rr(self, clock: _FakeClock) -> RemoteReranker:
        return RemoteReranker(
            "m", "https://rr.example.test/v1",
            breaker=guard.CircuitBreaker(clock=clock),
        )

    def test_repeated_fast_failures_stop_reaching_the_endpoint(self) -> None:
        rr = self._rr(_FakeClock())
        with _mock_http(_down, rr) as requests:
            for _ in range(10):
                _, applied, status = rr.rerank_with_status("q", _candidates())
                assert applied is False and status == "remote_unavailable"
        calls_while_closed = guard.FAILURE_THRESHOLD * 2  # one retry each
        assert len(requests) == calls_while_closed
        assert rr._breaker.state == guard.CircuitBreaker.OPEN

    def test_a_probe_runs_in_the_background_and_the_recall_does_not_wait(
        self,
    ) -> None:
        clock = _FakeClock()
        rr = self._rr(clock)
        with _mock_http(_down, rr):
            for _ in range(guard.FAILURE_THRESHOLD):
                rr.rerank_with_status("q", _candidates())
        assert rr._breaker.state == guard.CircuitBreaker.OPEN
        clock.now += guard.BASE_COOLDOWN_S + 1
        gate = threading.Event()

        def _slow_ok(request: httpx.Request) -> httpx.Response:
            gate.wait(5)
            return _ok(request)

        with _mock_http(_slow_ok, rr) as requests:
            t0 = time.monotonic()
            _, applied, _ = rr.rerank_with_status("q", _candidates())
            assert time.monotonic() - t0 < 0.1, "the recall waited on the probe"
            assert applied is False
            assert _wait_for(lambda: len(requests) == 1), "no probe was sent"
            gate.set()
            assert _wait_for(
                lambda: rr._breaker.state == guard.CircuitBreaker.TRIAL,
            )
            # The trial: the next real recall goes through and closes it.
            results, applied, status = rr.rerank_with_status("q", _candidates())
        assert applied is True and status == "applied"
        assert rr._breaker.state == guard.CircuitBreaker.CLOSED
        assert [f.fact_id for f, _ in results] == ["f2", "f1", "f0"]

    def test_a_failed_trial_backs_off_for_longer(self) -> None:
        clock = _FakeClock()
        rr = self._rr(clock)
        with _mock_http(_down, rr):
            for _ in range(guard.FAILURE_THRESHOLD):
                rr.rerank_with_status("q", _candidates())
        first = rr._breaker.cooldown_s
        clock.now += first + 1
        with _mock_http(_ok, rr):
            rr.rerank_with_status("q", _candidates())  # starts the probe
            assert _wait_for(
                lambda: rr._breaker.state == guard.CircuitBreaker.TRIAL,
            )
        with _mock_http(_down, rr):
            rr.rerank_with_status("q", _candidates())  # the trial fails
        assert rr._breaker.state == guard.CircuitBreaker.OPEN
        assert rr._breaker.cooldown_s == pytest.approx(first * 2)

    def test_only_one_probe_at_a_time(self) -> None:
        clock = _FakeClock()
        rr = self._rr(clock)
        with _mock_http(_down, rr):
            for _ in range(guard.FAILURE_THRESHOLD):
                rr.rerank_with_status("q", _candidates())
        clock.now += guard.BASE_COOLDOWN_S + 1
        gate = threading.Event()

        def _held(request: httpx.Request) -> httpx.Response:
            gate.wait(5)
            raise httpx.ConnectError("still down")

        with _mock_http(_held, rr) as requests:
            for _ in range(20):
                rr.rerank_with_status("q", _candidates())
            time.sleep(0.1)
            gate.set()
            assert _wait_for(lambda: not rr._breaker._probing)
        assert len(requests) <= 2, f"{len(requests)} probes for one cooldown"


# ---------------------------------------------------------------------------
# Unchanged behaviour: healthy endpoint, and no endpoint at all
# ---------------------------------------------------------------------------

class TestUnchangedBehaviour:

    def test_a_healthy_endpoint_reranks_every_recall_with_one_request(
        self,
    ) -> None:
        rr = RemoteReranker("m", "https://rr.example.test/v1")
        with _mock_http(_ok, rr) as requests:
            for _ in range(50):
                results, applied, status = rr.rerank_with_status(
                    "q", _candidates(),
                )
                assert applied is True and status == "applied"
                assert [f.fact_id for f, _ in results] == ["f2", "f1", "f0"]
        assert len(requests) == 50
        assert rr._breaker.state == guard.CircuitBreaker.CLOSED

    def test_concurrent_recalls_on_a_healthy_endpoint_are_all_reranked(
        self,
    ) -> None:
        """Several agents recalling at once must not lose reranking to the
        bound on requests in flight."""
        rr = RemoteReranker("m", "https://rr.example.test/v1")

        def _slowish(request: httpx.Request) -> httpx.Response:
            time.sleep(0.1)
            return _ok(request)

        n = 12
        barrier = threading.Barrier(n)
        outcomes: list[tuple[bool, str]] = []
        lock = threading.Lock()

        def recall() -> None:
            barrier.wait()
            _, applied, status = rr.rerank_with_status("q", _candidates())
            with lock:
                outcomes.append((applied, status))

        with _mock_http(_slowish, rr):
            threads = [threading.Thread(target=recall) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
        assert outcomes == [(True, "applied")] * n, outcomes

    def test_one_failure_among_successes_never_pauses_reranking(self) -> None:
        rr = RemoteReranker("m", "https://rr.example.test/v1")
        flaky = {"n": 0}

        def _sometimes(request: httpx.Request) -> httpx.Response:
            flaky["n"] += 1
            if flaky["n"] % 7 == 0:
                return httpx.Response(400, text="bad")
            return _ok(request)

        applied_count = 0
        with _mock_http(_sometimes, rr):
            for _ in range(40):
                _, applied, _ = rr.rerank_with_status("q", _candidates())
                applied_count += applied
        assert rr._breaker.state == guard.CircuitBreaker.CLOSED
        assert applied_count >= 34

    def test_no_endpoint_configured_builds_no_remote_reranker(self) -> None:
        from types import SimpleNamespace

        from superlocalmemory.core.engine_wiring import init_reranker

        cfg = SimpleNamespace(
            cross_encoder_backend="", cross_encoder_endpoint="",
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            trust_plain_http_lan=True,
        )
        with patch(
            "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
        ) as local:
            built = init_reranker(cfg)
        assert not isinstance(built, RemoteReranker)
        assert local.called


# ---------------------------------------------------------------------------
# The guard on its own
# ---------------------------------------------------------------------------

class TestCallWithin:

    def test_returns_the_value(self) -> None:
        slots = threading.BoundedSemaphore(2)
        assert guard.call_within(lambda _d: 7, 1.0, slots) == 7

    def test_reraises_the_error_in_the_caller(self) -> None:
        def boom(_d: float) -> None:
            raise ValueError("x")

        with pytest.raises(ValueError):
            guard.call_within(boom, 1.0, threading.BoundedSemaphore(1))

    def test_abandons_a_slow_call_and_frees_its_slot_when_it_ends(self) -> None:
        slots = threading.BoundedSemaphore(1)
        release = threading.Event()
        with pytest.raises(guard.DeadlineExceeded):
            guard.call_within(lambda _d: release.wait(5), 0.1, slots)
        with pytest.raises(guard.TooManyInFlight):
            guard.call_within(lambda _d: 1, 0.1, slots)
        release.set()
        assert _wait_for(lambda: _slot_free(slots))

    def test_the_callee_sees_the_absolute_deadline(self) -> None:
        seen: list[float] = []
        t0 = time.monotonic()
        guard.call_within(seen.append, 0.5, threading.BoundedSemaphore(1))
        assert t0 + 0.4 < seen[0] <= time.monotonic() + 0.5


def _slot_free(slots: threading.BoundedSemaphore) -> bool:
    if slots.acquire(blocking=False):
        slots.release()
        return True
    return False
