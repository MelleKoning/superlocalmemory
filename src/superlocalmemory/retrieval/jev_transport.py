# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""How the hosted answer check talks to its provider: one total deadline per
request, one shared client, and nothing sent once it is switched off.

Three properties, each one a bug before this module existed:

* **One wall-clock deadline.** httpx's ``timeout`` is per phase (connect,
  write, each read, pool), so a provider or proxy that trickles its reply kept
  a recall waiting far past its budget, and DNS resolution is not bounded by it
  at all. Here the request runs on a small worker pool and the caller waits for
  it until the deadline and not a moment longer; the worker also checks the
  deadline between chunks of the reply, so an abandoned request ends shortly
  after it. A request abandoned after it was sent may still be billed by the
  provider; one never started (deadline passed while queued) is never sent.
* **Nothing sent after shutdown.** Switching the check off, or withdrawing
  consent, calls ``close()``. Every request passes a gate under one lock just
  before it is sent; after ``close()`` the gate refuses, so a recall that was
  still loading its key or redacting its memories sends nothing.
* **One client.** httpx documents ``Client`` as safe to share between threads
  ("It can be shared between threads", httpx 0.28.1 ``Client`` docstring; its
  connection pool mutates state only under a thread lock). What was not safe
  was building it lazily without a lock: two first recalls at once built two
  clients and one was never closed. It is built under the gate's lock now, and
  closed by whichever of ``close()`` or the last in-flight request comes second.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import threading
import time
from typing import Any

import httpx

from superlocalmemory.core.outbound_http import GatedClient

logger = logging.getLogger(__name__)

#: Concurrent hosted requests per judge. One recall makes at most one, so this
#: bounds threads under a burst of recalls against a provider that hangs.
MAX_CONCURRENT_REQUESTS = 4
#: A reply larger than this is refused rather than held in memory. A real
#: answer for thirty memories is a few kilobytes.
MAX_REPLY_BYTES = 1_000_000

#: Failure codes. They name the kind of failure only — never the key, the
#: question or a memory.
FAILED_TIMEOUT = "timeout"
FAILED_SHUT_DOWN = "shut_down"
FAILED_TOO_LARGE = "too_large"


class HostedTransport:
    """POSTs to one provider. ``post`` returns ``(body, status_code, failure)``:
    the reply bytes on a 200, else None with a short failure code."""

    def __init__(self, *, transport: Any = None,
                 max_concurrent: int = MAX_CONCURRENT_REQUESTS) -> None:
        self._transport = transport
        self._max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._client: GatedClient | None = None
        self._executor: futures.ThreadPoolExecutor | None = None
        self._closed = False
        self._inflight = 0

    @property
    def closed(self) -> bool:
        return self._closed

    # -- the one call ------------------------------------------------------

    def post(self, endpoint: str, headers: dict[str, str], *, deadline: float,
             **payload: Any) -> tuple[bytes | None, int, str]:
        if deadline - time.monotonic() <= 0:
            return None, 0, FAILED_TIMEOUT
        executor = self._executor_or_none()
        if executor is None:
            return None, 0, FAILED_SHUT_DOWN
        try:
            future = executor.submit(self._send, endpoint, headers, deadline, payload)
        except RuntimeError:  # the executor shut down between the two lines above
            return None, 0, FAILED_SHUT_DOWN
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic()))
        except futures.TimeoutError:
            future.cancel()  # still queued: it will never be sent
            return None, 0, FAILED_TIMEOUT
        except futures.CancelledError:
            return None, 0, FAILED_SHUT_DOWN
        except Exception as exc:  # noqa: BLE001 — never the message: it may quote a body
            return None, 0, f"transport:{type(exc).__name__}"

    def _send(self, endpoint: str, headers: dict[str, str], deadline: float,
              payload: dict[str, Any]) -> tuple[bytes | None, int, str]:
        if deadline - time.monotonic() <= 0:
            return None, 0, FAILED_TIMEOUT  # waited in the queue too long: never sent
        client = self._begin()
        if client is None:
            return None, 0, FAILED_SHUT_DOWN
        try:
            return self._exchange(client, endpoint, headers, deadline, payload)
        except httpx.TimeoutException:
            return None, 0, FAILED_TIMEOUT
        except httpx.HTTPError as exc:
            return None, 0, f"transport:{type(exc).__name__}"
        except RuntimeError:  # the client was closed underneath this request
            return None, 0, FAILED_SHUT_DOWN
        finally:
            self._end()

    @staticmethod
    def _exchange(client: GatedClient, endpoint: str, headers: dict[str, str],
                  deadline: float, payload: dict[str, Any]) -> tuple[bytes | None, int, str]:
        remaining = max(0.01, deadline - time.monotonic())
        with client.stream("POST", endpoint, headers=headers,
                           timeout=httpx.Timeout(remaining), **payload) as response:
            if response.status_code != 200:
                return None, response.status_code, f"http_{response.status_code}"
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_REPLY_BYTES:
                    return None, 200, FAILED_TOO_LARGE
                if time.monotonic() > deadline:
                    return None, 200, FAILED_TIMEOUT
            return b"".join(chunks), 200, ""

    # -- lifecycle -----------------------------------------------------------

    def _executor_or_none(self) -> futures.ThreadPoolExecutor | None:
        with self._lock:
            if self._closed:
                return None
            if self._executor is None:
                self._executor = futures.ThreadPoolExecutor(
                    max_workers=self._max_concurrent, thread_name_prefix="jev-judge")
            return self._executor

    def _begin(self) -> GatedClient | None:
        """The gate: the last point before a request is sent."""
        with self._lock:
            if self._closed:
                return None
            if self._client is None:
                # The outbound gate's client: the body (already redacted by
                # the judge) is screened again on the final URL.
                self._client = GatedClient(transport=self._transport)
            self._inflight += 1
            return self._client

    def _end(self) -> None:
        with self._lock:
            self._inflight -= 1
            client = None
            if self._closed and self._inflight == 0:
                client, self._client = self._client, None
        _close_quietly(client)

    def close(self) -> None:
        """Refuse every request from now on. Never waits for one in flight."""
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
            client = None
            if self._inflight == 0:
                client, self._client = self._client, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        _close_quietly(client)


def _close_quietly(client: GatedClient | None) -> None:
    if client is None:
        return
    try:
        client.close()
    except Exception:  # noqa: BLE001 — closing must never fail a switch
        logger.debug("jev transport: client close failed", exc_info=True)


__all__ = [
    "FAILED_SHUT_DOWN",
    "FAILED_TIMEOUT",
    "FAILED_TOO_LARGE",
    "HostedTransport",
    "MAX_CONCURRENT_REQUESTS",
    "MAX_REPLY_BYTES",
]
