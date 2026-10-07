# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Consolidation finds a new fact's neighbours without reading every fact.

On a 21,847-fact store the consolidator read and decoded every fact twice per
new fact (merge candidates, then graph edges): about 7.7 s each time, which
held the background writer's lease for tens of seconds per memory, stalled
saves and kept new memories from being enriched. It now asks recall's exact
nearest-neighbour source. These tests pin that the answer is the same set the
full scan gave, that withheld facts never come back, and that no full read
happens. Synthetic vectors only.
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding.consolidator import MemoryConsolidator
from superlocalmemory.retrieval.canonical_vector_index import (
    CanonicalVectorIndex,
    candidate_vector_source,
    shared_index,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

DIM = 32


@pytest.fixture()
def store(tmp_path):
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    rng = np.random.default_rng(7)
    base = rng.normal(size=DIM)
    ids = []
    for i in range(240):
        # a third are near the base direction, so many pass the 0.5 cut
        v = base + rng.normal(scale=0.6 if i % 3 == 0 else 3.0, size=DIM)
        mid = db.store_memory(MemoryRecord(profile_id="default", content=f"note {i}"))
        ids.append(db.store_fact(AtomicFact(
            profile_id="default", memory_id=mid, content=f"synthetic note {i}",
            fact_type=FactType.SEMANTIC, embedding=[float(x) for x in v])))
    probe = AtomicFact(profile_id="default", memory_id="m-probe", content="probe",
                       fact_type=FactType.SEMANTIC,
                       embedding=[float(x) for x in base + rng.normal(scale=0.3, size=DIM)])
    return db, ids, probe


def _consolidator(db, vectors=None) -> MemoryConsolidator:
    c = MemoryConsolidator(db, embedder=object(), config=EncodingConfig())
    if vectors is not None:
        c.use_vector_source(vectors)
    return c


@pytest.mark.parametrize("k", [1, 5, 25])
def test_the_index_returns_exactly_what_the_full_scan_returned(store, k) -> None:
    db, _ids, probe = store
    old = _consolidator(db)._similar(probe, "default", k, set())
    new = _consolidator(db, CanonicalVectorIndex(db, DIM))._similar(probe, "default", k, set())
    assert [f.fact_id for f, _ in new] == [f.fact_id for f, _ in old]
    assert [round(s, 6) for _, s in new] == [round(s, 6) for _, s in old]
    assert len(old) == k  # the cut really applied


def test_excluded_and_withheld_facts_never_come_back(store) -> None:
    db, _ids, probe = store
    index = CanonicalVectorIndex(db, DIM)
    top = [f.fact_id for f, _ in _consolidator(db, index)._similar(probe, "default", 5, set())]
    db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (top[0],))
    again = _consolidator(db, index)._similar(probe, "default", 5, {top[1]})
    got = [f.fact_id for f, _ in again]
    assert top[0] not in got and top[1] not in got and len(got) == 5


def test_no_full_read_happens_with_a_vector_source(store, monkeypatch) -> None:
    db, _ids, probe = store
    index = CanonicalVectorIndex(db, DIM)
    index.search(probe.embedding, top_k=1, profile_id="default")  # build once (start-up)

    def boom(*_a, **_k):
        raise AssertionError("read every fact")

    monkeypatch.setattr(db, "get_all_facts", boom)
    c = _consolidator(db, index)
    assert c._similar(probe, "default", 5, set())
    c._create_semantic_edges(probe, "default")
    c._find_candidates(probe, "default")


def test_recall_and_consolidation_share_one_index(store) -> None:
    db, _ids, _probe = store
    a = candidate_vector_source(db, None, DIM)
    b = candidate_vector_source(db, None, DIM)
    assert a is b is shared_index(db, DIM)
