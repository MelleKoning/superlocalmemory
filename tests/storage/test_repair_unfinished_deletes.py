# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair`` finishes what an unfinished delete left behind, the safe way.

A delete whose writer timed out leaves the memory stored but unfindable (its
tombstone written, its search entries removed), and the person was told the
delete did not happen. The repair makes such a memory findable again. It never
does that for an unfinished erasure of a person or a profile: that is reported
so the erasure is run again.
"""

from __future__ import annotations

import uuid

from tests.core.test_stale_delete_tombstone import _WORDS, _age, _half_deleted, _state, _store


def _repair(engine, *, with_engine: bool = True) -> dict:
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    return Repair(engine._db.db_path, limits=Limits(pause_s=0.0, confirm_s=0.0),
                  engine=engine if with_engine else None).apply()


def _old(engine, fact_id: str) -> None:
    from superlocalmemory.core.delete_refusal import STALE_AFTER_S

    _age(engine, fact_id, STALE_AFTER_S + 1)


def test_an_unfinished_delete_is_made_findable_again(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    kept = _store(engine, _WORDS)
    healthy = _state(engine, kept)
    _half_deleted(engine, kept)
    young = _repair(engine)
    assert "unfinished_deletes.restored" not in young["done"], young["done"]
    assert _state(engine, kept)["tombstone"] == 1  # may be a delete still running

    _old(engine, kept)
    result = _repair(engine)
    assert result["done"].get("unfinished_deletes.restored") == 1, result["done"]
    assert _state(engine, kept) == {**healthy, "open_erasures": 0}
    again = _repair(engine)  # idempotent: nothing left to do
    assert not any(k.startswith("unfinished_deletes") for k in again["done"]), again["done"]


def test_without_the_engine_it_is_counted_not_changed(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    kept = _store(engine, _WORDS)
    _half_deleted(engine, kept)
    _old(engine, kept)
    broken = _state(engine, kept)
    result = _repair(engine, with_engine=False)
    assert result["done"].get("unfinished_deletes.needs_slm_running") == 1, result["done"]
    assert _state(engine, kept) == broken


def test_an_unfinished_erasure_of_a_person_is_never_undone(engine_with_mock_deps) -> None:
    from superlocalmemory.core.transactions import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import build_erasure_service

    engine = engine_with_mock_deps
    erased = _store(engine, _WORDS)
    ctx = OperationContext(operation_id=uuid.uuid4().hex, profile_id=engine._profile_id,
                           subject_id="Quorvantel", fact_ids=(erased,))  # an entity erasure
    assert build_erasure_service(engine).remove(engine._db, ctx).tombstoned
    _old(engine, erased)
    broken = _state(engine, erased)
    result = _repair(engine)
    assert result["done"].get("unfinished_deletes.erasure_to_run_again") == 1, result["done"]
    assert _state(engine, erased) == broken
    assert not broken["in_live_index"] and broken["bm25"] == 0
