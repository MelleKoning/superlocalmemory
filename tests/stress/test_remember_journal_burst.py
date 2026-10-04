# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A burst of concurrent saves is never refused by the admission journal.

128 remembers from 32 concurrent callers. Before the journal group-committed,
each save opened its own connections and queued on one lock for a separate
FULL-synchronous commit; under load a quarter of such a burst was refused
(503, nothing saved) and the journal prepare ran past its 2 s budget.

The contract: no save is refused, every save lands exactly once, the journal
prepare stays inside its budget (plus one commit's grace), and the
caller-facing acknowledgement stays inside the 1.5 s remember ceiling.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_SAVES = 128
_CALLERS = 32
_JOURNAL_DEADLINE_MS = 2_000
_ACCEPT_AFTER_MS = 1_200
_ACK_CEILING_SECONDS = 1.5
_COMMIT_GRACE_SECONDS = 0.05


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        for migration in (
            M018_ingestion_operations, M032_write_coordinator_admission,
            M033_projection_transactions, M034_obligation_integrity,
        ):
            migration.apply(conn)
    return db, CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=build_immediate_admission_handler(db, profile_id="default"),
        journal_path=tmp_path / "admission_journal.db",
        owner_id="journal-burst",
    )


def test_concurrent_burst_is_never_refused_and_lands_once(tmp_path, monkeypatch) -> None:
    from superlocalmemory.storage.admission_journal import (
        Actor,
        AdmissionJournal,
        RememberRequest,
    )

    db, runtime = _runtime(tmp_path, monkeypatch)
    prepare_seconds: list[float] = []
    ack_seconds: list[float] = []
    failures: list[str] = []
    lock = threading.Lock()
    original_prepare = AdmissionJournal.prepare

    def timed_prepare(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return original_prepare(self, *args, **kwargs)
        finally:
            with lock:
                prepare_seconds.append(time.monotonic() - started)

    monkeypatch.setattr(AdmissionJournal, "prepare", timed_prepare)
    actor = Actor("burst-actor", frozenset({"default"}), frozenset({"personal"}))

    def save(sequence: int) -> None:
        started = time.monotonic()
        try:
            receipt = runtime.remember(
                RememberRequest(
                    content=f"Burst fact {sequence}: the night ferry carries mail.",
                    profile_id="default",
                    source_type="stress-burst",
                    idempotency_key=f"journal-burst:{sequence}",
                    trusted_actor_id="burst-actor",
                ),
                actor,
                deadline_ms=_JOURNAL_DEADLINE_MS,
                accept_after_ms=_ACCEPT_AFTER_MS,
            )
            assert receipt.payload["status"] in {"queryable", "accepted"}
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            with lock:
                failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            with lock:
                ack_seconds.append(time.monotonic() - started)

    runtime.start()
    try:
        with ThreadPoolExecutor(max_workers=_CALLERS) as pool:
            list(pool.map(save, range(_SAVES)))
        assert runtime.wait_for_deferred(timeout=60.0)
    finally:
        runtime.stop()

    ordered = sorted(ack_seconds)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    print(
        f"\nburst: prepare max {max(prepare_seconds) * 1000:.0f} ms; "
        f"ack p95 {p95 * 1000:.0f} ms, max {ordered[-1] * 1000:.0f} ms"
    )
    assert failures == []
    assert max(prepare_seconds) <= _JOURNAL_DEADLINE_MS / 1000 + _COMMIT_GRACE_SECONDS
    assert p95 <= _ACK_CEILING_SECONDS
    assert db.execute("SELECT COUNT(*) AS n FROM atomic_facts")[0]["n"] == _SAVES
    assert db.execute("SELECT COUNT(*) AS n FROM write_commits")[0]["n"] == _SAVES
