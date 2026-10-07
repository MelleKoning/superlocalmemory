# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A model switch can pause background work so its swap is not starved.

Measured on a 22k-fact store copy: the pending materializer holds an operation
lease for each memory it materializes, one materialization outlasted the 5 s
drain timeout, and 51 activation attempts in a row timed out while each held
every request for 5 s. While ``pausing_background`` is held, background units
take no new lease; requests are still admitted.
"""

from __future__ import annotations

from superlocalmemory.server import unified_daemon
from superlocalmemory.server.profile_runtime import ProfileRuntime


def test_background_units_take_no_lease_while_paused_and_requests_still_do():
    runtime = ProfileRuntime("default")
    ran: list[str] = []
    with runtime.pausing_background():
        assert runtime.background_paused
        assert runtime._try_acquire_operation_nowait() is None
        result = unified_daemon._run_materializer_operation(
            runtime, lambda: object(), lambda engine: ran.append("ran"))
        assert result is None and ran == [], "a background unit ran while paused"
        with runtime.operation():  # a request is admitted as usual
            assert runtime.active_operations == 1
    assert not runtime.background_paused
    unified_daemon._run_materializer_operation(
        runtime, lambda: type("E", (), {"_profile_id": "default"})(),
        lambda engine: ran.append("ran"))
    assert ran == ["ran"]


def test_the_materializer_waits_while_paused_instead_of_spinning(monkeypatch):
    """With old queued saves present, a paused materializer re-read the queue
    in a tight loop for the whole pause (each item deferred at once)."""
    import time

    from superlocalmemory.cli import pending_store

    reads: list[int] = []
    monkeypatch.setattr(pending_store, "get_pending",
                        lambda **kw: reads.append(1) or [{"id": 1, "content": "x"}])
    monkeypatch.setattr(pending_store, "mark_failed", lambda *a, **k: None)
    runtime = ProfileRuntime("default")
    monkeypatch.setattr(unified_daemon, "_engine", object())
    monkeypatch.setattr(unified_daemon, "_profile_runtime", runtime)
    with runtime.pausing_background():
        unified_daemon._start_pending_materializer()
        try:
            time.sleep(1.0)
        finally:
            assert unified_daemon._stop_pending_materializer(timeout=5.0)
    assert len(reads) <= 2, f"the paused materializer read the queue {len(reads)} times in 1 s"
