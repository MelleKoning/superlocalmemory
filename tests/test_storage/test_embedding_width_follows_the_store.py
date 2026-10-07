# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The vector decoder expects the width the store holds, not a hard-coded 768.

With 768 hard-coded, every read of a store of any other width (a 384- or
1536-wide model, or any store after switching to one) logged a "truncated
write" warning per fact: 852,628 lines in one run on a 22k-fact store copy,
and recall's median rose to 21 s right after a switch to a 384-wide model.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from superlocalmemory.storage import embedding_codec as codec
from superlocalmemory.storage.embedding_projection import decode_embedding_array


@pytest.fixture(autouse=True)
def _restore_width():
    yield
    codec.set_expected_dimension(768)


def _blob(n: int) -> bytes:
    return np.ones(n, dtype=np.float32).tobytes()


def test_vectors_of_the_live_width_decode_without_a_warning(caplog):
    codec.set_expected_dimension(384)
    with caplog.at_level(logging.WARNING, logger=codec.logger.name):
        for i in range(50):
            assert len(codec.decode_embedding(_blob(384), fact_id=f"f{i}")) == 384
            assert decode_embedding_array(_blob(384), fact_id=f"f{i}").shape == (384,)
    assert not [r for r in caplog.records if "not the expected" in r.getMessage()]


def test_a_truncated_vector_is_still_reported_once_per_length(caplog):
    codec.set_expected_dimension(384)
    with caplog.at_level(logging.WARNING, logger=codec.logger.name):
        for i in range(20):
            codec.decode_embedding(_blob(383), fact_id=f"t{i}")
    warned = [r for r in caplog.records if "not the expected" in r.getMessage()]
    assert len(warned) == 1 and "383 floats" in warned[0].getMessage()


def test_an_engine_sets_the_width_of_its_live_space(tmp_path, monkeypatch):
    from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
    from superlocalmemory.core.embedding_live import bind_live_space

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    config = SLMConfig.load(tmp_path / "config.json")
    config.embedding = EmbeddingConfig(model_name="tiny", dimension=384)

    class _NoStore:
        def execute(self, sql, params=()):
            if "FROM atomic_facts" in sql:
                return []
            raise RuntimeError("no such table: embedding_space")

    codec.set_expected_dimension(768)
    bind_live_space(config, _NoStore())
    assert codec.expected_bytes() == 384 * 4
