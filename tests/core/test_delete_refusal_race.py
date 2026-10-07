# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A delete refused at its last check leaves the memory findable.

``delete_fact_authorized`` asks whether correction history protects the fact,
then removes the fact's search entries, then deletes the fact in one transaction
that asks again. A person can propose a correction naming the fact in between.
The last check then refuses, rightly, but the search entries were already gone:
the memory stayed stored and could no longer be found by its words or meaning.

These tests put the person's proposal exactly in that gap, on both delete paths
(the direct one and the sole-writer one), and require that after the refusal
everything search-visible is back and no erasure is left half-open.
"""

from __future__ import annotations

import uuid

import pytest

_WORDS = "Quorvantel stores the synthetic copper compass in the attic chest."


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


def _person_proposes(engine, predecessor: str, successor: str) -> None:
    """A correction a person proposed: never overtaken by a delete."""
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
            "successor_fact_id, reason_code, status, version, idempotency_key, "
            "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, "
            "created_at, updated_at) VALUES (?, ?, 'personal', ?, ?, 'user_edit', "
            "'proposed', 1, ?, 'person', 'user', 'high', 't', 't')",
            ("case-race", engine._profile_id, predecessor, successor, uuid.uuid4().hex))
        conn.commit()


def _inject_between_checks(monkeypatch, engine, keep: str, other: str) -> list[int]:
    """The first protection check passes; the proposal lands before the second."""
    from superlocalmemory.core import correction_protection

    real = correction_protection.blocking_cases
    calls: list[int] = []

    def racing(db, profile_id, fact_id):
        calls.append(1)
        if len(calls) == 2:
            _person_proposes(engine, keep, other)
        return real(db, profile_id, fact_id)

    monkeypatch.setattr(correction_protection, "blocking_cases", racing)
    return calls


def _bm25_index(engine):
    return getattr(getattr(engine, "_retrieval_engine", None), "_bm25", None)


def _search_state(engine, fact_id: str) -> dict:
    """Everything a search can reach the fact through, plus erasure leftovers."""
    pid = engine._profile_id
    index = _bm25_index(engine)
    vector_store = getattr(engine, "_vector_store", None)
    ann = getattr(engine, "_ann_index", None)
    return {
        "fact": _n(engine, "SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?", (fact_id,)),
        "bm25_rows": _n(engine, "SELECT COUNT(*) AS n FROM bm25_tokens WHERE fact_id = ?",
                        (fact_id,)),
        "bm25_live": None if index is None else fact_id in index._fact_id_set,
        "temporal": _n(engine, "SELECT COUNT(*) AS n FROM fact_temporal_validity "
                       "WHERE fact_id = ?", (fact_id,)),
        "vector": (None if vector_store is None or not getattr(vector_store, "available", False)
                   else fact_id in vector_store.indexed_fact_ids(pid)),
        "ann": None if ann is None or not hasattr(ann, "contains") else bool(ann.contains(fact_id)),
        "entities": _n(engine, "SELECT COUNT(*) AS n FROM fact_entity_associations "
                       "WHERE fact_id = ?", (fact_id,)),
        "tombstone": _n(engine, "SELECT COUNT(*) AS n FROM projection_tombstones "
                        "WHERE profile_id = ? AND fact_id = ?", (pid, fact_id)),
        "open_erasures": _n(engine, "SELECT COUNT(*) AS n FROM projection_obligations "
                            "WHERE kind = 'erase' AND profile_id = ?", (pid,)),
        "receipts": _n(engine, "SELECT COUNT(*) AS n FROM erasure_receipts"),
    }


def _found_by_words(engine, fact_id: str) -> bool:
    index = _bm25_index(engine)
    hits = index.search("copper compass attic", engine._profile_id, top_k=10)
    return fact_id in {h[0] for h in hits}


class _SoleWriter:
    """Runs the sole writer's own delete step in one transaction, as the writer does."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def delete_fact(self, profile_id, fact_id, *, idempotency_key=None):
        from superlocalmemory.core.remember_runtime import _delete_fact

        with self._engine._db.transaction():
            return _delete_fact(self._engine._db, fact_id, profile_id)


@pytest.mark.parametrize("path", ["direct", "sole_writer"])
def test_a_delete_refused_at_its_last_check_leaves_the_memory_findable(
    engine_with_mock_deps, monkeypatch, path,
):
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    keep = _store(engine, _WORDS)
    other = _store(engine, "Brellith walks the synthetic hound along the canal at dawn.")
    before = _search_state(engine, keep)
    assert before["fact"] == 1 and before["bm25_rows"] == 1, before
    assert before["tombstone"] == 0 and before["open_erasures"] == 0, before
    assert _found_by_words(engine, keep)

    calls = _inject_between_checks(monkeypatch, engine, keep, other)
    runtime = _SoleWriter(engine) if path == "sole_writer" else None
    with pytest.raises(CanonicalMutationConflict) as refused:
        delete_fact_authorized(engine, keep, trusted_actor_id=_actor(),
                               source_agent_id="test", canonical_runtime=runtime)

    assert len(calls) == 2, "the proposal must land between the two checks"
    assert "case-race (proposed)" in str(refused.value)
    assert "Nothing was changed" in str(refused.value)
    assert _search_state(engine, keep) == before
    assert _found_by_words(engine, keep)


def test_a_second_delete_after_the_race_is_refused_up_front(engine_with_mock_deps, monkeypatch):
    """After the restore, the next attempt is refused by the early check, not mid-way."""
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    keep = _store(engine, _WORDS)
    other = _store(engine, "Brellith walks the synthetic hound along the canal at dawn.")
    calls = _inject_between_checks(monkeypatch, engine, keep, other)
    with pytest.raises(CanonicalMutationConflict):
        delete_fact_authorized(engine, keep, trusted_actor_id=_actor(), source_agent_id="test")
    settled = _search_state(engine, keep)

    calls.clear()
    with pytest.raises(CanonicalMutationConflict):
        delete_fact_authorized(engine, keep, trusted_actor_id=_actor(), source_agent_id="test")
    assert len(calls) == 1, "refused by the first check: nothing was touched"
    assert _search_state(engine, keep) == settled
