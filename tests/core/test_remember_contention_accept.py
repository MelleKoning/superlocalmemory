# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A remember that meets a busy canonical writer is accepted, never refused.

Contention is injected deterministically: a second connection holds the
``memory.db`` write lock (``BEGIN IMMEDIATE``) for as long as the test wants,
which is exactly what a long enrichment, maintenance or legacy write does to
the coordinator under machine load. Before the fix every remember in that
window came back as ``CanonicalRememberUnavailable`` (HTTP 503) even though the
request was already durable in the admission journal.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from superlocalmemory.core.remember_runtime import (
    CanonicalRememberRuntime,
    CanonicalRememberUnavailable,
)
from superlocalmemory.storage.admission_journal import Actor, RememberRequest

_ACTOR = "contention-accept-daemon"
_SHORT_DEADLINE_MS = 300


def _build(data_dir: Path, owner: str = "contention-runtime") -> CanonicalRememberRuntime:
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    db = DatabaseManager(data_dir / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        for migration in (
            M018_ingestion_operations,
            M032_write_coordinator_admission,
            M033_projection_transactions,
            M034_obligation_integrity,
        ):
            migration.apply(conn)
        conn.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other', 'other')"
        )
    return CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=build_immediate_admission_handler(db, profile_id="default"),
        journal_path=data_dir / "admission_journal.db",
        owner_id=owner,
    )


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "slm-contention"
    path.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(path))
    return path


@pytest.fixture()
def runtime(data_dir: Path):
    rt = _build(data_dir)
    rt.start()
    yield rt
    rt.stop()


class _HeldWriteLock:
    """Hold memory.db's single SQLite write lock until released."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), timeout=5, isolation_level=None)
        self._conn.execute("BEGIN IMMEDIATE")

    def release(self) -> None:
        if self._conn is not None:
            self._conn.execute("ROLLBACK")
            self._conn.close()
            self._conn = None


def _request(key: str, *, profile: str = "default", text: str | None = None) -> RememberRequest:
    return RememberRequest(
        content=text or f"Contention fact {key}: the harbour pilot rotates on Tuesdays.",
        profile_id=profile,
        source_type="contention-test",
        idempotency_key=key,
        trusted_actor_id=_ACTOR,
    )


def _actor(profile: str = "default") -> Actor:
    return Actor(_ACTOR, frozenset({profile}), frozenset({"personal"}))


def _count(db_path: Path, sql: str, params: tuple = ()) -> int:
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        return int(conn.execute(sql, params).fetchone()[0])
    finally:
        conn.close()


def test_remember_under_held_write_lock_is_accepted_not_refused(runtime, data_dir):
    db_path = data_dir / "memory.db"
    lock = _HeldWriteLock(db_path)
    try:
        started = time.monotonic()
        receipt = runtime.remember(
            _request("held-1"), _actor(), deadline_ms=_SHORT_DEADLINE_MS,
        )
        elapsed = time.monotonic() - started
        payload = receipt.payload
        # The truth: durable, not yet searchable.
        assert payload["status"] == "accepted"
        assert payload["durable"] is True
        assert payload["queryable"] is False
        assert payload["fact_ids"] == []
        assert payload["idempotency_key"] == "held-1"
        assert payload["admission_id"]
        assert elapsed < _SHORT_DEADLINE_MS / 1000 + 0.5
        assert _count(db_path, "SELECT COUNT(*) FROM atomic_facts") == 0
    finally:
        lock.release()
    assert runtime.wait_for_deferred(timeout=10.0)
    assert _count(db_path, "SELECT COUNT(*) FROM atomic_facts") == 1
    assert _count(db_path, "SELECT COUNT(*) FROM write_commits") == 1
    assert runtime.journal.get(payload["admission_id"]).state == "committed"
    # The same key now returns the canonical receipt.
    final = runtime.remember(_request("held-1"), _actor(), deadline_ms=2_000)
    assert final.payload["status"] == "queryable"
    assert len(final.payload["fact_ids"]) == 1


def test_resending_the_same_key_while_deferred_never_duplicates(runtime, data_dir):
    db_path = data_dir / "memory.db"
    lock = _HeldWriteLock(db_path)
    try:
        first = runtime.remember(_request("dup-1"), _actor(), deadline_ms=_SHORT_DEADLINE_MS)
        second = runtime.remember(_request("dup-1"), _actor(), deadline_ms=_SHORT_DEADLINE_MS)
        assert first.payload["status"] == second.payload["status"] == "accepted"
        assert first.payload["admission_id"] == second.payload["admission_id"]
    finally:
        lock.release()
    third = runtime.remember(_request("dup-1"), _actor(), deadline_ms=2_000)
    assert runtime.wait_for_deferred(timeout=10.0)
    assert third.payload["status"] == "queryable"
    assert _count(db_path, "SELECT COUNT(*) FROM atomic_facts") == 1
    assert _count(db_path, "SELECT COUNT(*) FROM write_commits") == 1


def test_same_key_in_two_profiles_is_two_memories(runtime, data_dir):
    db_path = data_dir / "memory.db"
    lock = _HeldWriteLock(db_path)
    try:
        a = runtime.remember(
            _request("shared-key", text="Profile default owns the lantern rota."),
            _actor(), deadline_ms=_SHORT_DEADLINE_MS,
        )
        b = runtime.remember(
            _request("shared-key", profile="other", text="Profile other owns the bell rota."),
            _actor("other"), deadline_ms=_SHORT_DEADLINE_MS,
        )
        assert a.payload["admission_id"] != b.payload["admission_id"]
    finally:
        lock.release()
    assert runtime.wait_for_deferred(timeout=10.0)
    assert _count(
        db_path, "SELECT COUNT(*) FROM atomic_facts WHERE profile_id='default'",
    ) == 1
    assert _count(
        db_path, "SELECT COUNT(*) FROM atomic_facts WHERE profile_id='other'",
    ) == 1


def test_crash_between_accept_and_commit_replays_exactly_once(data_dir):
    db_path = data_dir / "memory.db"
    first = _build(data_dir, owner="before-crash")
    first.start()
    lock = _HeldWriteLock(db_path)
    try:
        receipt = first.remember(_request("crash-1"), _actor(), deadline_ms=_SHORT_DEADLINE_MS)
        assert receipt.payload["status"] == "accepted"
        # The daemon goes away while the writer is still blocked.
        first.stop()
    finally:
        lock.release()
    assert _count(db_path, "SELECT COUNT(*) FROM atomic_facts") == 0

    second = _build(data_dir, owner="after-crash")
    second.start()  # replays the journal before publishing readiness
    try:
        assert _count(db_path, "SELECT COUNT(*) FROM atomic_facts") == 1
        resend = second.remember(_request("crash-1"), _actor(), deadline_ms=2_000)
        assert resend.payload["status"] == "queryable"
        assert _count(db_path, "SELECT COUNT(*) FROM atomic_facts") == 1
        assert _count(db_path, "SELECT COUNT(*) FROM write_commits") == 1
    finally:
        second.stop()


def test_a_broken_writer_is_still_refused_not_claimed_saved(runtime, monkeypatch):
    from superlocalmemory.storage.write_coordinator import WriteCoordinatorError

    def broken(*_args, **_kwargs):
        raise WriteCoordinatorError("canonical write command was rejected")

    monkeypatch.setattr(runtime.coordinator, "submit", broken)
    with pytest.raises(CanonicalRememberUnavailable):
        runtime.remember(_request("broken-1"), _actor(), deadline_ms=_SHORT_DEADLINE_MS)
    assert runtime.deferred_count == 0


def test_acknowledgement_latency_under_contention_within_ceiling(runtime, data_dir):
    from superlocalmemory.server.unified_daemon import (
        _REMEMBER_ADMISSION_DEADLINE_MS,
        _REMEMBER_JOURNAL_DEADLINE_MS,
        _REMEMBER_TOTAL_CEILING_SECONDS,
    )

    db_path = data_dir / "memory.db"
    samples: list[float] = []
    lock = _HeldWriteLock(db_path)
    try:
        for index in range(12):
            started = time.monotonic()
            receipt = runtime.remember(
                _request(f"lat-{index}"), _actor(),
                deadline_ms=_REMEMBER_JOURNAL_DEADLINE_MS,
                accept_after_ms=_REMEMBER_ADMISSION_DEADLINE_MS,
            )
            samples.append(time.monotonic() - started)
            assert receipt.payload["status"] == "accepted"
    finally:
        lock.release()
    samples.sort()
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print(f"\nack latency under held lock: p50={samples[len(samples)//2]:.3f}s "
          f"p95={p95:.3f}s max={samples[-1]:.3f}s")
    assert p95 <= _REMEMBER_TOTAL_CEILING_SECONDS
    assert runtime.wait_for_deferred(timeout=20.0)
    assert _count(db_path, "SELECT COUNT(*) FROM atomic_facts") == 12


def test_accepted_save_for_a_deleted_profile_ends_with_a_recorded_failure(
    runtime, data_dir,
):
    """A profile deleted before its accepted save commits: one failure, recorded.

    Retrying can never succeed (a deleted profile does not come back), so the
    committer must stop, record why in the journal, and not retry forever.
    """
    db_path = data_dir / "memory.db"
    lock = _HeldWriteLock(db_path)
    try:
        receipt = runtime.remember(
            _request("deleted-profile-1", profile="other"), _actor("other"),
            deadline_ms=_SHORT_DEADLINE_MS,
        )
        assert receipt.payload["status"] == "accepted"
        lock._conn.execute("DELETE FROM profiles WHERE profile_id='other'")
        lock._conn.execute("COMMIT")
        lock._conn.close()
        lock._conn = None
    finally:
        lock.release()
    assert runtime.wait_for_deferred(timeout=10.0), "the committer kept retrying"
    entry = runtime.journal.get(receipt.payload["admission_id"])
    assert entry.state == "rejected"
    assert entry.error_code == "UNKNOWN_PROFILE"
    assert _count(db_path, "SELECT COUNT(*) FROM write_commits") == 0
