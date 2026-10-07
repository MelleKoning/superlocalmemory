# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``/health`` reads the database in a worker thread, never on the event loop.

It ran three SQLite counts (and the projection depth) inside an ``async`` route,
on the loop every request shares. Sampled on a copy of a 22k-fact store while
the daemon warmed up: the loop sat in those counts 4.6-5.1 s of an 8 s window,
so finished recalls could not send their answers. Clients poll ``/health``
hardest exactly then, waiting for readiness.
"""

from __future__ import annotations

import asyncio
import threading


def _blocker(result):
    """A slow read that can only finish early if the event loop is free meanwhile."""
    entered, loop_ran, seen = threading.Event(), threading.Event(), []

    def slow(*_args):
        entered.set()
        seen.append(loop_ran.wait(timeout=10.0))
        return result

    async def drive(coro):
        task = asyncio.ensure_future(coro)
        while not entered.is_set() and not task.done():
            await asyncio.sleep(0.005)
        loop_ran.set()  # reached only while ``slow`` waits, if the loop is free
        return await task

    return slow, drive, seen


def _health_route(app):
    return next(route for route in app.routes if getattr(route, "path", None) == "/health")


def test_the_loop_keeps_running_while_health_reads(tmp_path, monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    slow_counts, drive, seen = _blocker({"dead_letter_count": 0})
    monkeypatch.setattr(unified_daemon, "_ops_failure_counts", slow_counts)
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.migration_result = {"applied": [], "skipped": [], "failed": [], "details": {}}

    payload = asyncio.run(drive(_health_route(app).endpoint()))
    assert seen == [True], "the health route held the event loop while it read the database"
    assert payload["dead_letter_count"] == 0


def test_status_counts_do_not_hold_the_loop_either(tmp_path, monkeypatch) -> None:
    """``/status`` counted every fact and edge on the loop (a cold count: seconds)."""
    from types import SimpleNamespace

    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    slow_counts, drive, seen = _blocker({})
    monkeypatch.setattr(unified_daemon, "_ops_failure_counts", slow_counts)
    app = unified_daemon.create_app()
    app.state.engine = None
    app.state.daemon_descriptor = SimpleNamespace(port=1)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/status")

    payload = asyncio.run(drive(route.endpoint()))
    assert seen == [True], "the status route held the event loop while it read the database"
    assert payload["status"] == "running"


def test_a_finished_recall_builds_its_answer_off_the_loop(monkeypatch) -> None:
    """The recall's answer (memory text read, serialisation) was built on the loop."""
    from types import SimpleNamespace

    from superlocalmemory.server import recall_core

    slow_envelope, drive, seen = _blocker({"ok": True})
    monkeypatch.setattr(recall_core, "_envelope", slow_envelope)
    monkeypatch.setattr(recall_core, "_engine_call", lambda engine, call: object())
    monkeypatch.setattr("superlocalmemory.server.profile_runtime.get_profile_runtime",
                        lambda state: SimpleNamespace(snapshot=None))
    call = recall_core.RecallCall(query="q", limit=1, session_id="s", agent_id="a", fast=True)

    assert asyncio.run(drive(recall_core.run_recall(object(), call, app_state=None))) == {"ok": True}
    assert seen == [True], "the recall's answer was built on the event loop"
