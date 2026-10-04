# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon owns the only Answer Check history writer: started once, stopped
with a final save, and absent when the history is switched off."""

from __future__ import annotations

import inspect
import sqlite3
import threading

import pytest

from superlocalmemory.core import answer_check_history as h
from superlocalmemory.core import answer_check_history_store as store
from superlocalmemory.server import unified_daemon
from superlocalmemory.storage import migration_runner as mr

from ..test_core.test_answer_check_history import make_response


@pytest.fixture(autouse=True)
def fresh():
    store._reset_for_testing()
    h._reset_for_testing()
    yield
    store._reset_for_testing()
    h._reset_for_testing()


def _writers() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "slm-answer-check-history"]


def test_start_is_idempotent_and_stop_flushes(tmp_path) -> None:
    from superlocalmemory.storage import schema

    learning, memory = tmp_path / "learning.db", tmp_path / "memory.db"
    with sqlite3.connect(memory) as conn:
        schema.create_all_tables(conn)
    mr.apply_all(learning, memory)
    store.start_writer(learning)
    store.start_writer(learning)
    assert len(_writers()) == 1 and h.is_enabled()
    for _ in range(7):
        h.record_recall_verdict(make_response(), profile_id="default")
    assert store.stop_writer() == 0
    assert _writers() == [] and not h.is_enabled()
    with sqlite3.connect(learning) as conn:
        assert conn.execute("SELECT COUNT(*) FROM answer_check_events").fetchone()[0] == 7


def test_lifespan_wiring() -> None:
    """Started after the outcome queue, gated on the setting; stopped on shutdown."""
    src = inspect.getsource(unified_daemon)
    start = src.index("from superlocalmemory.core.answer_check_history_store import start_writer")
    gate = src.rindex('getattr(_rc, "answer_check_history", True) is not False', 0, start)
    assert start - gate < 200
    assert src.index("start_worker(_memory_db)") < start
    assert "stop_writer(timeout_s=2.0)" in src
    # GET reads gated by the RBAC middleware. The gate list moved to
    # server/read_gates.py in 627e5d6a; assert the behaviour, not the source.
    for path in ("/api/v3/answer-check/history", "/api/v3/answer-check/history/live",
                 "/api/v3/answer-check/history/summary"):
        assert unified_daemon._is_sensitive_dashboard_read("GET", path), path
    assert "include_router(answer_check_history_router)" in src


def test_other_processes_never_record() -> None:
    """No writer in this process -> recording is a no-op (CLI fallback, workers)."""
    for _ in range(10):
        h.record_recall_verdict(make_response(), profile_id="default")
    assert h.counters()["recorded"] == 0
