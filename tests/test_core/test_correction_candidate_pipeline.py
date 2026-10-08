"""Canonical write-path contract for automatic correction candidates."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from superlocalmemory.core.store_pipeline import _record_correction_candidate
from superlocalmemory.storage.migrations import M042_correction_case_ledger as m042


class _CoordinatorBoundDatabase:
    """Tiny test double: its transaction is owned by the caller, not the helper."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @contextmanager
    def raw_connection(self):
        yield self._connection


def test_candidate_uses_the_current_transaction_and_carries_no_detector_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as conn:
        m042.apply(conn)

    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _record_correction_candidate(
            _CoordinatorBoundDatabase(conn),
            operation_id="ingest-1",
            profile_id="alpha",
            scope="project",
            predecessor_fact_id="old-release",
            successor_fact_id="new-release",
            reason_code="temporal_contradiction",
            trusted_actor_id="host-actor-1",
        )
        stored = conn.execute(
            "SELECT reason_code, proposed_by_actor_id FROM correction_cases"
        ).fetchone()
        assert stored == ("temporal_contradiction", "host-actor-1")
        assert conn.execute("SELECT COUNT(*) FROM correction_events").fetchone() == (1,)
        conn.execute("ROLLBACK")
    finally:
        conn.close()

    with sqlite3.connect(path) as verify:
        assert verify.execute("SELECT COUNT(*) FROM correction_cases").fetchone() == (0,)
        assert verify.execute("SELECT COUNT(*) FROM correction_events").fetchone() == (0,)


def test_candidate_refuses_to_self_attest_when_trusted_actor_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as conn:
        m042.apply(conn)

    with sqlite3.connect(path) as conn:
        _record_correction_candidate(
            _CoordinatorBoundDatabase(conn),
            operation_id="ingest-1",
            profile_id="alpha",
            scope="personal",
            predecessor_fact_id="old-release",
            successor_fact_id="new-release",
            reason_code="consolidation_supersede",
            trusted_actor_id="",
        )
        assert conn.execute("SELECT COUNT(*) FROM correction_cases").fetchone() == (0,)


def _propose(conn: sqlite3.Connection, *, operation_id: str, successor: str,
             reason: str = "temporal_contradiction") -> None:
    _record_correction_candidate(
        _CoordinatorBoundDatabase(conn),
        operation_id=operation_id,
        profile_id="alpha",
        scope="project",
        predecessor_fact_id="old-release",
        successor_fact_id=successor,
        reason_code=reason,
        trusted_actor_id="host-actor-1",
    )


def _ledger(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as conn:
        m042.apply(conn)
    return sqlite3.connect(path)


def test_second_candidate_for_the_same_memory_is_quiet_and_keeps_the_open_one(
    tmp_path: Path, caplog,
) -> None:
    """The ledger keeps one open case per memory. When two saves (or two facts
    of one save) both suspect the same older memory, the case already open
    stays: it may be under review, and a case is history that is never
    replaced. The repeat is not an error and must not log at warning level."""
    import logging

    conn = _ledger(tmp_path)
    try:
        _propose(conn, operation_id="ingest-1", successor="new-a")
        with caplog.at_level(logging.DEBUG):
            _propose(conn, operation_id="ingest-2", successor="new-b",
                     reason="consolidation_supersede")
        rows = conn.execute(
            "SELECT successor_fact_id, reason_code, status FROM correction_cases"
        ).fetchall()
        assert rows == [("new-a", "temporal_contradiction", "proposed")]
        assert conn.execute("SELECT COUNT(*) FROM correction_events").fetchone() == (1,)
        loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert loud == []
    finally:
        conn.close()


def test_repeat_of_the_same_candidate_is_quiet(tmp_path: Path, caplog) -> None:
    import logging

    conn = _ledger(tmp_path)
    try:
        _propose(conn, operation_id="ingest-1", successor="new-a")
        with caplog.at_level(logging.DEBUG):
            _propose(conn, operation_id="ingest-3", successor="new-a")
        assert conn.execute("SELECT COUNT(*) FROM correction_cases").fetchone() == (1,)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    finally:
        conn.close()


def test_a_decided_open_case_is_never_displaced_by_a_new_candidate(tmp_path: Path) -> None:
    """An applied case is a decision. A later guess about the same memory
    leaves it exactly as it is."""
    conn = _ledger(tmp_path)
    try:
        _propose(conn, operation_id="ingest-1", successor="new-a")
        conn.execute("UPDATE correction_cases SET status='applied', version=1")
        _propose(conn, operation_id="ingest-2", successor="new-b")
        assert conn.execute(
            "SELECT successor_fact_id, status FROM correction_cases"
        ).fetchall() == [("new-a", "applied")]
    finally:
        conn.close()


def test_a_closed_case_does_not_block_a_fresh_candidate(tmp_path: Path) -> None:
    """A rejected or rolled-back case is no longer open, so the memory can be
    suspected again."""
    conn = _ledger(tmp_path)
    try:
        _propose(conn, operation_id="ingest-1", successor="new-a")
        conn.execute("UPDATE correction_cases SET status='rejected', version=1")
        _propose(conn, operation_id="ingest-2", successor="new-b")
        assert sorted(conn.execute(
            "SELECT successor_fact_id, status FROM correction_cases"
        ).fetchall()) == [("new-a", "rejected"), ("new-b", "proposed")]
    finally:
        conn.close()
