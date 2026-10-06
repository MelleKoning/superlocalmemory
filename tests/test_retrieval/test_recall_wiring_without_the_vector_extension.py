# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A recall on a store without the vector extension never re-reads every fact.

Simulates a Python that cannot load SQLite extensions (no vector store) on the
real retrieval wiring (``init_retrieval``), then recalls: no channel may fall
back to reading the whole fact table, and every meaning-based channel must
answer instead of timing out. On 4.1.21 the semantic channel and the
spreading-activation seeds each did that full read on every recall.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.engine_wiring import init_retrieval
from superlocalmemory.core.modes import Mode
from superlocalmemory.retrieval.canonical_vector_index import CanonicalVectorIndex
from superlocalmemory.storage.database import DatabaseManager
from tests.test_retrieval.cross_scope_fixture import REQ, build_store


@pytest.fixture()
def wired(tmp_path):
    store = build_store(tmp_path / "w.db")
    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.retrieval.use_cross_encoder = False
    query = store.queries[0].tolist()
    embedder = SimpleNamespace(embed=lambda text: query, is_ready=True)
    engine = init_retrieval(config, store.db, embedder, None, None, vector_store=None)
    yield store, engine, query
    engine.close()


def test_the_meaning_channels_share_the_in_memory_index(wired) -> None:
    _, engine, _ = wired
    source = engine._semantic._vector_store
    assert isinstance(source, CanonicalVectorIndex)
    assert engine._spreading_activation._vector_store is source
    assert engine._hopfield is None or engine._hopfield._vector_store is source
    assert engine._kind_vectors is source and engine._kind_membership is not None


def test_a_recall_never_reads_the_whole_fact_table(wired, monkeypatch) -> None:
    store, engine, query = wired
    warm = getattr(engine._semantic._vector_store, "warm", None)
    if warm is not None:
        warm(REQ)  # the daemon does this at start

    def whole_table(*a, **k):
        raise AssertionError("a channel read every fact")

    monkeypatch.setattr(DatabaseManager, "get_all_facts", whole_table)
    response = engine.recall("content", REQ, limit=10)
    status = response.channel_status
    assert status["semantic"] == "ok", status
    assert status["spreading_activation"] in ("ok", "empty"), status
    assert "timeout" not in status.values(), status
    assert response.results
