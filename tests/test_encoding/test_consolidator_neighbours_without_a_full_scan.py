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


class _NotYetIndexed:
    """A vector source that has not indexed some facts yet, as during a save:
    the facts a save stores get their vectors only once it commits."""

    def __init__(self, inner, hidden: set[str]) -> None:
        self._inner, self._hidden = inner, hidden

    def search(self, embedding, top_k: int, profile_id: str):
        return [h for h in self._inner.search(embedding, top_k=top_k, profile_id=profile_id)
                if h[0] not in self._hidden]


def test_facts_stored_by_the_same_save_are_still_neighbours(store) -> None:
    db, _ids, probe = store
    old = [f.fact_id for f, _ in _consolidator(db)._similar(probe, "default", 5, set())]
    hidden = set(old[:2])
    source = _NotYetIndexed(CanonicalVectorIndex(db, DIM), hidden)
    new = _consolidator(db, source)._similar(probe, "default", 5, set(), tuple(hidden))
    assert [f.fact_id for f, _ in new] == old


def test_a_memorys_own_facts_are_linked_when_it_is_saved(tmp_path) -> None:
    """Through a real save: the prefixed fact and the copy without its
    "[name] " prefix are one memory's facts and must be linked, as the full
    scan linked them (4.1.22: they were not, which moved recall ranking)."""
    import hashlib
    import re
    from unittest.mock import MagicMock, patch

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id
    from superlocalmemory.storage.models import Mode

    def embed(text: str) -> list[float]:
        vec = np.zeros(768, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % 768] += 1.0
        return (vec / float(np.linalg.norm(vec))).tolist()

    embedder = MagicMock()
    embedder.embed.side_effect = embed
    embedder.embed_batch.side_effect = lambda texts: [embed(t) for t in texts]
    embedder.is_available = True
    embedder.compute_fisher_params.return_value = ([0.0] * 768, [1.0] * 768)
    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.retrieval.use_cross_encoder = False
    engine = MemoryEngine(config)
    with patch("superlocalmemory.core.engine_wiring.init_embedder", return_value=embedder):
        engine.initialize()
        engine._embedder = embedder
    try:
        canonical_store(engine, "[newsletter] session ended 2026-10-04 17:02 | recent: fix typo "
                        "in invoice email footer", source_type="python-api",
                        trusted_actor_id=local_trusted_actor_id("python-api"),
                        metadata={"project": "newsletter"}, require_complete=True)
        facts = {r[0]: r[1] for r in engine._db.execute(
            "SELECT fact_id, content FROM atomic_facts")}
        assert len(facts) >= 2, facts
        linked = {frozenset((r[0], r[1])) for r in engine._db.execute(
            "SELECT source_id, target_id FROM graph_edges WHERE edge_type = 'semantic'")}
        prefixed = [f for f, c in facts.items() if c.startswith("[newsletter]")]
        stripped = [f for f, c in facts.items() if c.startswith("session ended")]
        assert prefixed and stripped
        assert frozenset((prefixed[0], stripped[0])) in linked
    finally:
        engine.close()
