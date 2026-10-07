# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A correction cannot be proposed or reviewed while its memory is being deleted.

A delete writes the memory's erasure tombstone and removes its search entries,
then deletes it in a transaction that asks again whether correction history
protects it. A correction proposed in between used to turn the delete into a
late refusal (the safety net in core/delete_refusal.py then put the search
entries back). Now the window is closed where it opens: while the tombstone
exists, proposing a correction naming the memory, or applying one, is refused
as a conflict that says why, and the delete completes with nothing left behind.

These run through the real sole writer (CanonicalRememberRuntime), so the
refusal is checked as callers see it: a conflict, not "temporarily unavailable".
"""

from __future__ import annotations

import uuid

import pytest

from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
    M042_correction_case_ledger,
)

_WORDS = "Quorvantel stores the synthetic copper compass in the attic chest."
_OTHER = "Brellith walks the synthetic hound along the canal at dawn."


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _store(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store

    receipt = canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                              require_complete=True, return_receipt=True)
    return list(receipt.final_fact_ids)[0]


def _n(engine, sql: str, args: tuple = ()) -> int:
    return int(dict(engine._db.execute(sql, args)[0])["n"])


@pytest.fixture
def runtime(engine_with_mock_deps):
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        M042_correction_case_ledger.apply(conn)
    rt = CanonicalRememberRuntime.for_engine(engine_with_mock_deps)
    rt.start()
    try:
        yield rt
    finally:
        rt.stop()


def _machine_case(engine, predecessor: str, successor: str) -> str:
    """A pending proposal only SLM made: the user's delete overtakes it."""
    case_id = "case-machine-" + uuid.uuid4().hex[:8]
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
            "successor_fact_id, reason_code, status, version, idempotency_key, "
            "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, "
            "created_at, updated_at) VALUES (?, ?, 'personal', ?, ?, 'consolidation_update', "
            "'proposed', 0, ?, 'slm', 'host_attested', 'canonical_writer', 't', 't')",
            (case_id, engine._profile_id, predecessor, successor, uuid.uuid4().hex))
        conn.commit()
    return case_id


def _during_delete(monkeypatch, action):
    """Run ``action`` once, after the tombstone is written and before the final check."""
    from superlocalmemory.storage import erasure_fence

    real, seen = erasure_fence.mark_erasing, []

    def hooked(profile_id, fact_id):
        if not seen:
            seen.append(action())
        return real(profile_id, fact_id)

    monkeypatch.setattr(erasure_fence, "mark_erasing", hooked)
    return seen


def _attempt(call):
    try:
        call()
    except Exception as exc:  # the test asserts on what came back
        return exc
    return None


def _left_behind(engine, fact_id: str) -> dict:
    pid = engine._profile_id
    return {
        "fact": _n(engine, "SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?", (fact_id,)),
        "bm25": _n(engine, "SELECT COUNT(*) AS n FROM bm25_tokens WHERE fact_id = ?", (fact_id,)),
        "cases": _n(engine, "SELECT COUNT(*) AS n FROM correction_cases WHERE "
                    "predecessor_fact_id = ? OR successor_fact_id = ?", (fact_id, fact_id)),
        "open_erasures": _n(engine, "SELECT COUNT(*) AS n FROM projection_obligations WHERE "
                            "kind = 'erase' AND profile_id = ? AND state NOT IN "
                            "('erased', 'verified')", (pid,)),
    }


def test_a_person_cannot_propose_a_correction_of_a_memory_being_deleted(
    engine_with_mock_deps, runtime, monkeypatch,
):
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    doomed = _store(engine, _WORDS)
    seen = _during_delete(monkeypatch, lambda: _attempt(lambda: runtime.create_correction_successor(
        engine._profile_id, doomed, "succ" + doomed[:12],
        "Quorvantel stores the copper compass in the cellar.",
        trusted_actor_id="person-test", idempotency_key="edit-during-delete")))

    result = delete_fact_authorized(engine, doomed, trusted_actor_id=_actor(),
                                    source_agent_id="test", canonical_runtime=runtime)

    assert len(seen) == 1, "the edit must run inside the delete window"
    refused = seen[0]
    assert isinstance(refused, CanonicalMutationConflict), repr(refused)
    assert "is being deleted" in str(refused) and "nothing was changed" in str(refused)
    assert result.get("ok") is True and result.get("erasure_verified") is True, result
    assert _left_behind(engine, doomed) == {"fact": 0, "bm25": 0, "cases": 0,
                                            "open_erasures": 0}
    assert _n(engine, "SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?",
              ("succ" + doomed[:12],)) == 0


def test_a_machine_correction_cannot_be_applied_while_its_memory_is_deleted(
    engine_with_mock_deps, runtime, monkeypatch,
):
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    doomed, other = _store(engine, _WORDS), _store(engine, _OTHER)
    case_id = _machine_case(engine, other, doomed)
    seen = _during_delete(monkeypatch, lambda: _attempt(lambda: runtime.transition_correction(
        engine._profile_id, case_id, action="apply", expected_version=0,
        actor_id="person-test", idempotency_key="apply-during-delete")))

    result = delete_fact_authorized(engine, doomed, trusted_actor_id=_actor(),
                                    source_agent_id="test", canonical_runtime=runtime)

    assert len(seen) == 1
    assert isinstance(seen[0], CanonicalMutationConflict), repr(seen[0])
    assert "is being deleted" in str(seen[0])
    # The delete went through and overtook the still-pending machine case.
    assert result.get("ok") is True, result
    assert _left_behind(engine, doomed)["fact"] == 0
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases WHERE case_id = ?",
              (case_id,)) == 0
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases_overtaken WHERE case_id = ?",
              (case_id,)) == 1
    assert _n(engine, "SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?", (other,)) == 1


def test_a_stale_review_is_a_conflict_not_an_outage(engine_with_mock_deps, runtime):
    """Pre-existing: a review against an old case version was reported as the
    writer being "temporarily unavailable", so clients retried it forever."""
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    first, other = _store(engine, _WORDS), _store(engine, _OTHER)
    case_id = _machine_case(engine, other, first)
    with pytest.raises(CanonicalMutationConflict, match="case state changed"):
        runtime.transition_correction(engine._profile_id, case_id, action="apply",
                                      expected_version=7, actor_id="person-test",
                                      idempotency_key="stale-apply")
