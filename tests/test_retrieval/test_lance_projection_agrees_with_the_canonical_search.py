# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""The Lance projection must return the same candidates the canonical search does.

The projection worker puts every fact recall may return into Lance, ``cold`` and
``archived`` included, and stores the fact's lifecycle in a ``tier`` column. The
canonical search ranks all of them. The Lance candidate source used to ask only
for the ``active`` and ``warm`` tiers, so on a store where a sixth of the
memories are cold the two searches disagreed on almost every query: the sampled
check logged "Lance semantic projection diverged from SQLite" at start-up, and
the unsampled searches (49 of every 50) served Lance's answer, which had left the
cold memories out.

These tests build a real store and a real Lance projection of it, then compare
what the semantic channel returns through each.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from superlocalmemory.retrieval.semantic_channel import SemanticChannel
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryLifecycle, MemoryRecord

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        importlib.util.find_spec("lancedb") is None,
        reason="LanceDB optional dependency not installed",
    ),
]

DIM = 8
LIFECYCLES = (
    MemoryLifecycle.ACTIVE,
    MemoryLifecycle.WARM,
    MemoryLifecycle.COLD,
    MemoryLifecycle.ARCHIVED,
)


def _embedding(seed: int) -> list[float]:
    v = np.random.RandomState(seed).randn(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture()
def store(tmp_path: Path):
    """A store with facts in every lifecycle, and a Lance projection of it."""
    from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(real_schema)
    lance = LanceDBVectorBackend(str(tmp_path / "lance"), dimension=DIM)
    facts: list[AtomicFact] = []
    for seed in range(1, 21):
        lifecycle = LIFECYCLES[seed % len(LIFECYCLES)]
        record = MemoryRecord(profile_id="default", content=f"m{seed}", session_id="s")
        db.store_memory(record)
        fact = AtomicFact(
            profile_id="default", memory_id=record.memory_id,
            content=f"fact {seed}", embedding=_embedding(seed), lifecycle=lifecycle,
        )
        db.store_fact(fact)
        facts.append(fact)
    # What the projection worker writes: every visible fact, tier = lifecycle.
    lance.add_vectors(
        [f.fact_id for f in facts], [f.embedding for f in facts],
        [f.lifecycle.value for f in facts], "default",
    )
    yield db, lance, facts
    lance.close()


def _channels(db, lance):
    plain = SemanticChannel(db, vector_store=None)
    projected = SemanticChannel(db, vector_store=None)
    projected.set_scale_vector_backend(lance)
    return plain, projected


def test_lance_candidates_include_cold_and_archived_facts(store) -> None:
    db, lance, facts = store
    _, projected = _channels(db, lance)
    cold = [f for f in facts if f.lifecycle in (MemoryLifecycle.COLD, MemoryLifecycle.ARCHIVED)]
    assert cold, "the fixture must contain cold and archived facts"

    # Query on a cold fact's own vector: it must be its own nearest neighbour.
    target = cold[0]
    found = projected._search_via_lance(
        target.embedding, np.array(target.embedding, dtype=np.float32),
        "default", top_k=5,
    )
    assert target.fact_id in {fid for fid, _ in found}


def test_sampled_check_finds_no_divergence_on_a_mixed_lifecycle_store(store) -> None:
    db, lance, facts = store
    plain, projected = _channels(db, lance)

    for fact in facts:
        query = fact.embedding
        expected = {fid for fid, _ in plain.search(query, "default", top_k=8)}
        got = {fid for fid, _ in projected.search(query, "default", top_k=8)}
        assert got == expected

    # The first search is always the sampled one, so the check ran at least once
    telemetry = projected.scale_projection_telemetry()
    assert telemetry["shadow_checks"] >= 1
    assert telemetry["shadow_mismatches"] == 0
    assert telemetry["shadow_errors"] == 0


def test_the_projection_stays_correct_after_a_fact_changes_lifecycle(store) -> None:
    """The tier column is only a copy; a stale one must not change an answer."""
    db, lance, facts = store
    plain, projected = _channels(db, lance)
    mover = next(f for f in facts if f.lifecycle == MemoryLifecycle.WARM)
    # SQLite says the fact went cold; Lance still has it as warm.
    db.execute(
        "UPDATE atomic_facts SET lifecycle = 'cold' WHERE fact_id = ?",
        (mover.fact_id,),
    )
    other = next(f for f in facts if f.lifecycle == MemoryLifecycle.ACTIVE)
    db.execute(
        "UPDATE atomic_facts SET lifecycle = 'warm' WHERE fact_id = ?",
        (other.fact_id,),
    )
    for fact in (mover, other):
        expected = {fid for fid, _ in plain.search(fact.embedding, "default", top_k=8)}
        got = {fid for fid, _ in projected.search(fact.embedding, "default", top_k=8)}
        assert got == expected
    assert projected.scale_projection_telemetry()["shadow_mismatches"] == 0
