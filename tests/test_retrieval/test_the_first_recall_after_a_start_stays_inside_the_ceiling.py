# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The first recall after a start answers inside the 3.0 s ceiling (L3-M2).

On a fresh daemon the first recalls took 8.06 s: the recall waited the whole
channel hang guard (8 s) for a model that was still loading. The ceiling is
3.0 s total. The answer must come back inside it, say honestly that its vector
channels were warming and that it is incomplete, and say which stage used the
time. A model that loaded before is still waited for as in 4.1.19 -- see
test_a_reloading_model_is_waited_for.py.

Real engine, real ``EmbeddingService``; the model is a child process that takes
longer to load than the ceiling (tests/helpers/fake_embedding_worker.py).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService
from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval.answer_check_status import RECALL_CEILING_S
from superlocalmemory.server.recall_serializer import recall_response_metadata
from tests.helpers import fake_embedding_worker as fake

_MODEL_LOAD_SECONDS = 6.0   # longer than the whole ceiling


@pytest.fixture()
def cold_engine(mode_a_config, monkeypatch):
    from superlocalmemory.core.engine import MemoryEngine

    sp = fake.install(monkeypatch, fake.answering_worker(_MODEL_LOAD_SECONDS))
    svc = EmbeddingService(EmbeddingConfig(dimension=768))
    engine = MemoryEngine(mode_a_config)
    with patch("superlocalmemory.core.engine_wiring.init_embedder", return_value=svc):
        engine.initialize()
    engine._embedder = svc
    yield engine, svc
    engine.close()
    sp.reap()


def test_the_first_recall_is_inside_the_ceiling_and_says_it_is_warming(cold_engine) -> None:
    engine, svc = cold_engine
    engine.store("The quarterly planning offsite is in Lisbon this year.")
    assert svc.has_loaded_once is False

    t0 = time.monotonic()
    response = engine.recall("Where is the quarterly planning offsite?")
    total = time.monotonic() - t0

    assert total <= RECALL_CEILING_S, f"first recall took {total:.2f}s"
    # Honest: the vector channels did not run because the model was loading,
    # and the answer says it is incomplete rather than looking whole.
    assert response.channel_status.get("semantic") == chstat.WARMING
    assert "semantic" in response.incomplete_channels
    # The channels that need no vector still found the memory.
    assert any("Lisbon" in r.fact.content for r in response.results)
    # Which stage used the time travels with the answer.
    stages = response.stage_ms
    assert 900 <= stages["query_embedding"] <= 1600, stages
    assert stages["channels"] >= stages["query_embedding"]
    # The transport envelope carries the honest flags.
    meta = recall_response_metadata(response)
    assert "semantic" in meta["incomplete_channels"]
    assert meta["channel_status"]["semantic"] == chstat.WARMING


def test_a_warm_recall_reports_no_warming(mode_a_config, mock_embedder) -> None:
    """The ready path is unchanged and is not marked incomplete."""
    from superlocalmemory.core.engine import MemoryEngine

    engine = MemoryEngine(mode_a_config)
    with patch("superlocalmemory.core.engine_wiring.init_embedder",
               return_value=mock_embedder):
        engine.initialize()
    engine._embedder = mock_embedder
    try:
        engine.store("The quarterly planning offsite is in Lisbon this year.")
        response = engine.recall("Where is the quarterly planning offsite?")
        assert chstat.WARMING not in response.channel_status.values()
        assert "semantic" not in response.incomplete_channels
        assert "query_embedding" in response.stage_ms
    finally:
        engine.close()
