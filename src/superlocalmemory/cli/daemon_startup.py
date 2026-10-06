# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A bounded wait for a daemon that is still starting.

A fresh daemon needs seconds (a laptop with an empty store) to about a minute
(a small shared box loading models) before ``/health`` answers. Before 4.1.22
every daemon-backed call made in that window failed at once with
``DAEMON_UNAVAILABLE (daemon_unreachable)``, so the first ``remember`` after an
MCP host started SLM was lost to a race the caller could not see.

The rule here:

* wait only when there is evidence the daemon is starting — this process is
  spawning it, or the descriptor says ``starting`` and its process is alive;
* wait at most :func:`start_wait_budget` seconds (default 20 s);
* never spawn anything and never report success: when the budget runs out the
  caller gets its usual "unavailable" answer, which the diagnosis then names
  ``daemon_starting`` with a retry hint.

Why 20 s: the hosts that call SLM's MCP tools cancel a tool call at about 60 s
(Codex ``tool_timeout_sec`` defaults to 60; the Claude desktop app cancels
stdio tool calls near 60 s). A daemon request itself may take up to its own
30 s timeout, so 20 s of waiting plus 30 s of work stays about 10 s under the
host's limit, and it covers the 5-11 s cold start measured on a developer Mac.
``SLM_DAEMON_START_WAIT_S`` overrides it (clamped to 0-45 s; 0 disables).
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

DEFAULT_START_WAIT_S = 20.0
MAX_START_WAIT_S = 45.0
RETRY_HINT_S = 10
_POLL_S = 0.25
# A starting daemon binds its port before HTTP is up, so a 2 s health read
# would just hang; while the descriptor says "starting" one probe is short.
STARTING_PROBE_S = 0.5
# Back-to-back retries inside one tool call (remember tries three times, 50-
# 100 ms apart) must not each wait the full budget again: a wait that ran out
# this recently for the same daemon instance is not repeated.
_REWAIT_GRACE_S = 2.0

_state_lock = threading.Lock()
_spawning_count = 0
_spawn_finished = threading.Event()
_spawn_finished.set()
_expired_at: dict[str, float] = {}


def start_wait_budget(cap: float | None = None) -> float:
    """Seconds a call may wait for a starting daemon (never negative)."""
    raw = os.environ.get("SLM_DAEMON_START_WAIT_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_START_WAIT_S
    except ValueError:
        value = DEFAULT_START_WAIT_S
    if value != value:  # NaN
        value = DEFAULT_START_WAIT_S
    value = max(0.0, min(value, MAX_START_WAIT_S))
    if cap is not None:
        value = min(value, max(0.0, float(cap)))
    return value


@contextmanager
def spawning() -> Iterator[None]:
    """Mark that THIS process is spawning the daemon for the block's duration."""
    global _spawning_count
    with _state_lock:
        _spawning_count += 1
        _spawn_finished.clear()
    try:
        yield
    finally:
        with _state_lock:
            _spawning_count -= 1
            if _spawning_count <= 0:
                _spawning_count = 0
                _spawn_finished.set()


def this_process_is_spawning() -> bool:
    return not _spawn_finished.is_set()


def starting_descriptor() -> Any | None:
    """The descriptor of a live daemon that says it is still starting."""
    from superlocalmemory.cli import daemon as _d

    descriptor = _d.read_descriptor()
    if descriptor is None or getattr(descriptor, "state", "") != "starting":
        return None
    return descriptor if _d._descriptor_process_is_alive(descriptor) else None


def start_in_progress() -> bool:
    """Evidence that the daemon is starting right now."""
    return this_process_is_spawning() or starting_descriptor() is not None


def _recently_expired(instance_id: str) -> bool:
    with _state_lock:
        at = _expired_at.get(instance_id)
    return at is not None and time.monotonic() - at < _REWAIT_GRACE_S


def _mark_expired(instance_id: str) -> None:
    with _state_lock:
        _expired_at.clear()  # one daemon per root; keep the map tiny
        _expired_at[instance_id] = time.monotonic()


def probe_health(d: Any, port: int, remaining: float) -> dict | None:
    """One health probe that cannot outlive the wait's own deadline.

    A starting daemon reserves its port early, so a connect can succeed and
    then hang until the 2 s read limit; capping by ``remaining`` keeps the
    whole wait inside its budget.
    """
    limit = max(0.05, min(2.0, remaining))
    try:
        return d._fetch_health(port, timeout=limit)
    except TypeError:  # a stand-in that takes only the port
        return d._fetch_health(port)


def health_probe_timeout(descriptor: Any) -> float:
    """Health read limit for ``descriptor``: short while it is starting."""
    return STARTING_PROBE_S if getattr(descriptor, "state", "") == "starting" else 2.0


def wait_for_starting_daemon(
    *, cap: float | None = None, seconds: float | None = None,
) -> tuple[Any, dict] | None:
    """Wait (bounded) for a starting daemon; return ``(descriptor, health)``.

    Returns ``None`` at once when nothing shows a start in progress, so a
    stopped or wedged daemon behaves exactly as before. Returns ``None`` when
    the process exits or the budget runs out. Never spawns. ``seconds``
    replaces the configured budget (``slm serve stop`` waits longer than a
    tool call may).
    """
    from superlocalmemory.cli import daemon as _d

    first = starting_descriptor()
    if first is None and not this_process_is_spawning():
        return None
    if seconds is None and first is not None and _recently_expired(first.instance_id):
        return None
    budget = start_wait_budget(cap) if seconds is None else max(0.0, float(seconds))
    deadline = time.monotonic() + budget
    last_instance = getattr(first, "instance_id", "")
    while True:
        descriptor = _d.read_descriptor()
        if descriptor is not None:
            last_instance = descriptor.instance_id
            if not _d._descriptor_process_is_alive(descriptor):
                return None  # it exited: the diagnosis says so
            health = probe_health(_d, descriptor.port, deadline - time.monotonic())
            if health is not None and _d.descriptor_matches_health(descriptor, health):
                return descriptor, health
        elif not this_process_is_spawning():
            return None  # no descriptor and nobody here is starting one
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_instance:
                _mark_expired(last_instance)
            return None
        time.sleep(min(_POLL_S, remaining))


def starting_diagnosis(descriptor: Any) -> dict[str, str]:
    """The truthful answer for a call made while the daemon is starting."""
    try:
        age = max(0, int(time.time() - float(descriptor.process_create_time)))
        detail = f" (pid {descriptor.pid}, launched {age}s ago)"
    except (TypeError, ValueError, AttributeError):
        pid = getattr(descriptor, "pid", None)
        detail = f" (pid {pid})" if pid else ""
    return {
        "reason": "daemon_starting",
        "message": (
            f"SuperLocalMemory is still starting{detail}; "
            "a first start loads its models and can take up to a minute. "
            "This request was not sent."
        ),
        "hint": f"Retry in about {RETRY_HINT_S} seconds.",
    }
