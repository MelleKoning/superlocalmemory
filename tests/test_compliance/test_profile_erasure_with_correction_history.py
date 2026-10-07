# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Erasing a whole profile erases it even when a person corrected its memories.

The correction ledger (M042) refers to both facts of a case ``ON DELETE
RESTRICT``. A profile erasure deletes every fact of the profile, so a case it
leaves behind would stop the erasure partway.
"""

from __future__ import annotations

import sqlite3

_TEXT = "Brindlemoor keeps the synthetic pewter kettle on the third shelf."
_EDIT = "Brindlemoor keeps the synthetic pewter kettle on the fourth shelf."
_PROFILE = "erasable"


def _n(engine, sql: str, args: tuple = ()) -> int:
    try:
        return int(dict(engine._db.execute(sql, args)[0])["n"])
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return 0
        raise


def _corrected_profile(engine) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id
    from superlocalmemory.core.remember_runtime import _execute_mutation
    from superlocalmemory.storage.write_coordinator import CommandKind

    engine._db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                       (_PROFILE, _PROFILE))
    receipt = canonical_store(engine, _TEXT, source_type="python-api",
                              trusted_actor_id=local_trusted_actor_id("python-api"),
                              require_complete=True, return_receipt=True, profile_id=_PROFILE)
    fact_id = list(receipt.final_fact_ids)[0]
    with engine._db.raw_connection() as conn:
        _execute_mutation(engine._db, CommandKind.PROPOSE_CORRECTION, _PROFILE, {
            "fact_id": fact_id, "successor_fact_id": "f" * 16, "content": _EDIT,
            "trusted_actor_id": "person-test", "idempotency_key": "profile-erase-1"},
            connection=conn)
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases WHERE profile_id = ?",
              (_PROFILE,)) == 1
    return fact_id


def test_gdpr_profile_erasure_erases_a_corrected_profile(engine_with_mock_deps) -> None:
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    engine = engine_with_mock_deps
    _corrected_profile(engine)
    counts = GDPRCompliance(engine._db, engine=engine).forget_profile(_PROFILE)
    assert not counts.get("erasure_aborted"), counts
    for table in ("atomic_facts", "memories", "correction_cases"):
        assert _n(engine, f"SELECT COUNT(*) AS n FROM {table} WHERE profile_id = ?",
                  (_PROFILE,)) == 0, (table, counts)
