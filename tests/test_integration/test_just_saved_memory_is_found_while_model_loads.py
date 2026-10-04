# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Remember, then recall at once, while the embedding model is still loading.

The reported fresh-install defect (4.1.20 WP10): ``slm remember`` said
"Queryable", and ``slm recall`` straight after said "No confident match" for
about a minute. The contract this pins, on a fresh store through the real
engine: the memory is found, OR the answer says the search is incomplete —
never a confident "no match" from a search that did not run.
"""

from __future__ import annotations

import threading
import time

from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval import engine as retrieval_engine_mod


class _LoadingModel:
    """An embedder whose model is still loading (``is_warm`` is False)."""

    def __init__(self, hold: float) -> None:
        self.release = threading.Event()
        self._hold = hold

    is_warm = False
    is_available = True

    def embed(self, text):
        self.release.wait(self._hold)
        return [0.0] * 768


def _recall_with_loading_model(engine, monkeypatch, query: str):
    monkeypatch.setattr(retrieval_engine_mod, "CHANNEL_HANG_GUARD_SECONDS", 0.5)
    loading = _LoadingModel(hold=6.0)
    engine._retrieval_engine._embedder = loading
    try:
        t0 = time.monotonic()
        response = engine.recall(query, limit=5)
        return response, time.monotonic() - t0
    finally:
        loading.release.set()


def test_a_just_saved_memory_is_found_while_the_model_loads(
    engine_with_mock_deps, monkeypatch,
) -> None:
    engine = engine_with_mock_deps
    fact_ids = engine.store(
        "The zebra-quokka migration window for Project Halcyon is 14 March")
    assert fact_ids, "store returned no queryable fact"

    response, elapsed = _recall_with_loading_model(
        engine, monkeypatch, "When is the Halcyon migration window?")

    found = [r.fact.content for r in response.results]
    assert any("Halcyon" in c for c in found), (
        f"just-saved memory not found while the model loads: {found}, "
        f"status={response.channel_status}")
    assert elapsed < 4.0, f"recall waited {elapsed:.1f}s on a loading model"
    assert response.channel_status.get("semantic") == chstat.WARMING
    assert "semantic" in response.incomplete_channels


def test_nothing_found_while_loading_is_reported_incomplete(
    engine_with_mock_deps, monkeypatch,
) -> None:
    """A paraphrase only the vector channels could match is not found yet —
    and the answer must say why rather than claim the store has nothing."""
    engine = engine_with_mock_deps
    engine.store("The zebra-quokka migration window for Project Halcyon is 14 March")

    response, _ = _recall_with_loading_model(
        engine, monkeypatch, "Xylophone pterodactyl")

    # Whatever came back (temporal may surface the recent memory), the answer
    # must never present itself as a complete search.
    assert "semantic" in response.incomplete_channels, (
        "an answer from a partial search was reported as complete")
    assert all(chstat.is_fault(response.channel_status[c])
               for c in response.incomplete_channels)
