# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The user's action wins over a correction only a machine proposed (Varun, 2026-10-06).

A pending case SLM proposed by itself (consolidation or a date contradiction,
never reviewed) no longer blocks the user's delete, replace or edit: it is
closed as overtaken, recorded, and can be put back. A case a person proposed,
or one already applied, still blocks exactly as before.
"""

from __future__ import annotations

import json
import sqlite3

import pytest


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _fact(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store

    return list(canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                                require_complete=True, return_receipt=True).final_fact_ids)[0]


def _propose(engine, case_id: str, pred: str, succ: str, *, machine: bool,
             status: str = "proposed") -> None:
    """A case recorded exactly as each proposer records it today."""
    from superlocalmemory.core.remember_replaces import ledger_actor_id
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
        transition_on_connection,
    )

    actor = (CorrectionActor(actor_id="canonical-writer", actor_kind="host_attested",
                             trust_tier="canonical_writer") if machine else
             CorrectionActor(actor_id=ledger_actor_id(_actor()),
                             actor_kind="host_authenticated", trust_tier="trusted"))
    reason = "consolidation_supersede" if machine else "direct_content_correction"
    conn = sqlite3.connect(engine._db.db_path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    propose_on_connection(conn, case_id=case_id, profile_id=engine._profile_id,
                          scope="personal", predecessor_fact_id=pred, successor_fact_id=succ,
                          reason_code=reason, actor=actor, idempotency_key=f"k-{case_id}",
                          is_profile_active=lambda p: True, is_actor_trusted=lambda a: True)
    if status == "applied":
        transition_on_connection(conn, case_id=case_id, expected_version=0, actor=actor,
                                 operation_id=f"apply-{case_id}", from_status="proposed",
                                 to_status="applied", mutate_temporal=False,
                                 is_profile_active=lambda p: True,
                                 is_actor_trusted=lambda a: True)
    conn.execute("COMMIT")
    conn.close()


def _delete(engine, fact_id: str):
    from superlocalmemory.core.mutations import delete_fact_authorized

    return delete_fact_authorized(engine, fact_id, trusted_actor_id=_actor(),
                                  source_agent_id="test")


def _rows(engine, sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in engine._db.execute(sql, args)]


@pytest.fixture
def two(engine_with_mock_deps):
    engine = engine_with_mock_deps
    return engine, _fact(engine, "Synthetic heron count at the east pond was nine."), \
        _fact(engine, "Synthetic heron count at the east pond was eleven.")


def test_a_machine_proposal_no_longer_blocks_a_delete(two):
    engine, old, new = two
    _propose(engine, "case-m", old, new, machine=True)

    result = _delete(engine, old)

    assert result.get("ok") is True, result
    assert _rows(engine, "SELECT 1 FROM correction_cases WHERE case_id = 'case-m'") == []
    [closed] = _rows(engine, "SELECT * FROM correction_cases_overtaken WHERE case_id = 'case-m'")
    assert closed["closed_reason"] == "overtaken by a user action"
    assert closed["user_action"] == "delete" and closed["actor_id"] == _actor()
    assert json.loads(closed["case_json"])["status"] == "proposed"
    assert [e["event_type"] for e in json.loads(closed["events_json"])] == ["proposed"]
    assert _rows(engine, "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (new,))  # untouched


@pytest.mark.parametrize("machine,status", [(False, "proposed"), (True, "applied"),
                                            (False, "applied")])
def test_a_person_proposed_or_applied_case_still_blocks(two, machine, status):
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine, old, new = two
    _propose(engine, "case-h", old, new, machine=machine, status=status)

    with pytest.raises(CanonicalMutationConflict, match="protected by correction history"):
        _delete(engine, old)
    assert _rows(engine, "SELECT status FROM correction_cases WHERE case_id = 'case-h'") == [
        {"status": status}]


def test_overtake_then_restore_puts_the_case_back_exactly(two):
    from superlocalmemory.core import overtaken_cases as ot

    engine, old, new = two
    _propose(engine, "case-u", old, new, machine=True)
    before = _rows(engine, "SELECT * FROM correction_cases WHERE case_id = 'case-u'")
    events = _rows(engine, "SELECT * FROM correction_events WHERE case_id = 'case-u'")

    with engine._db.transaction():
        assert ot.overtake(engine._db, ot.cases_naming(engine._db, [old]),
                           user_action="replace", actor_id="u", operation_id="op") == ["case-u"]
    with engine._db.transaction():
        assert ot.restore(engine._db, "case-u") == {"restored": "case-u", "events": 1}

    assert _rows(engine, "SELECT * FROM correction_cases WHERE case_id = 'case-u'") == before
    assert _rows(engine, "SELECT * FROM correction_events WHERE case_id = 'case-u'") == events
    with pytest.raises(ot.RestoreRefused):
        ot.restore(engine._db, "case-u")  # once only


def test_restore_is_refused_once_the_user_deleted_the_fact(two):
    from superlocalmemory.core import overtaken_cases as ot

    engine, old, new = two
    _propose(engine, "case-d", old, new, machine=True)
    assert _delete(engine, old).get("ok") is True
    with pytest.raises(ot.RestoreRefused, match="no longer exists"):
        ot.restore(engine._db, "case-d")


def _replace(engine, replaces: str, successor: str) -> dict:
    from superlocalmemory.core.remember_replaces import apply_replacement

    conn = sqlite3.connect(engine._db.db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row  # as the canonical writer's connection
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        out = apply_replacement(conn, engine._profile_id, {
            "replaces": replaces, "successor_fact_id": successor,
            "trusted_actor_id": _actor()})
        conn.execute("COMMIT")
        return out
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def test_a_machine_proposal_no_longer_blocks_an_explicit_replace(two):
    engine, old, new = two
    third = _fact(engine, "Synthetic heron count at the east pond was twelve.")
    _propose(engine, "case-r", old, third, machine=True)

    result = _replace(engine, old, new)

    assert result["ok"] is True and result["fact_ids"] == [old]
    [closed] = _rows(engine, "SELECT user_action FROM correction_cases_overtaken "
                             "WHERE case_id = 'case-r'")
    assert closed["user_action"] == "replace"


def test_a_person_proposal_still_blocks_a_replace(two):
    from superlocalmemory.core.remember_replaces import ReplacementRefused

    engine, old, new = two
    third = _fact(engine, "Synthetic heron count at the east pond was twelve.")
    _propose(engine, "case-p", old, third, machine=False)

    with pytest.raises(ReplacementRefused, match="waiting for review"):
        _replace(engine, old, new)
    assert _rows(engine, "SELECT status FROM correction_cases WHERE case_id = 'case-p'") == [
        {"status": "proposed"}]
