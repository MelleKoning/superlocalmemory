"""Background embeds wait for a recall's query embedding, not for the whole recall.

Under steady recall traffic some recall is nearly always in flight (96% of
samples during an ingestion run on a 22k-fact store). Background embedding
waited for NO recall in flight, so a remember's inline embedding ran out its
budget every time and the enrichment backlog grew instead of draining. A recall
uses the embedder once, for its question; after that the worker is free while
the recall searches, reranks and is judged. A recall that has not got its
vector yet still comes first, and a plain ``begin_recall`` (no hold) keeps the
old rule for its whole duration.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

from superlocalmemory.core import recall_gate
from superlocalmemory.core.embeddings import EmbeddingService


@pytest.fixture(autouse=True)
def _clean_gate():
    assert recall_gate.in_flight() == 0
    yield
    assert recall_gate.in_flight() == 0
    assert recall_gate.recalls_needing_embedder() == 0


def _background_waiter(event: threading.Event) -> threading.Thread:
    def run() -> None:
        with recall_gate.background_work():
            recall_gate.wait_for_embedder_idle()
        event.set()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


@contextmanager
def _recall_in_flight():
    """A recall held the way recall_core holds it; yields a step runner on its engine thread."""
    hold = recall_gate.RecallHold()
    go, done = threading.Event(), threading.Event()
    steps: list = []

    def engine_work() -> None:
        while not done.is_set():
            if steps:
                steps.pop(0)()
                go.set()
            time.sleep(0.005)

    worker = threading.Thread(target=hold.run, args=(engine_work,), daemon=True)
    worker.start()

    def step(fn) -> None:
        go.clear()
        steps.append(fn)
        assert go.wait(2.0)

    try:
        yield step
    finally:
        done.set()
        worker.join(2.0)
        hold.leave()


def test_background_embed_waits_while_the_recall_has_no_vector():
    with _recall_in_flight():
        ready = threading.Event()
        _background_waiter(ready)
        assert not ready.wait(0.3)
        assert recall_gate.recalls_needing_embedder() == 1


def test_background_embed_proceeds_once_the_recall_has_its_vector():
    with _recall_in_flight() as step:
        ready = threading.Event()
        _background_waiter(ready)
        assert not ready.wait(0.2)
        step(recall_gate.mark_query_embedded)
        assert ready.wait(1.0), "background embed still blocked after the query was embedded"
        assert recall_gate.in_flight() == 1  # the recall itself is still running
    assert recall_gate.recalls_needing_embedder() == 0


def test_marking_twice_counts_once():
    with _recall_in_flight() as step:
        step(recall_gate.mark_query_embedded)
        step(recall_gate.mark_query_embedded)
        assert recall_gate.recalls_needing_embedder() == 0
        assert recall_gate.in_flight() == 1


def test_a_plain_begin_recall_keeps_background_embeds_waiting():
    recall_gate.begin_recall()
    try:
        recall_gate.mark_query_embedded()  # no hold on this thread: no effect
        ready = threading.Event()
        _background_waiter(ready)
        assert not ready.wait(0.3)
    finally:
        recall_gate.end_recall()


def test_embedder_request_lock_admits_background_work_after_the_query_vector():
    """The local embedder's own admission uses the narrower rule."""
    svc = EmbeddingService.__new__(EmbeddingService)
    svc._lock = threading.Lock()
    with _recall_in_flight() as step:
        step(recall_gate.mark_query_embedded)
        got = threading.Event()

        def background() -> None:
            with recall_gate.background_work():
                with svc._request_lock():
                    got.set()
        threading.Thread(target=background, daemon=True).start()
        assert got.wait(1.0), "embedder admission still waits for the whole recall"


def test_a_recalls_query_embedding_marks_it():
    """Retrieval marks the recall as soon as its question is embedded."""
    from superlocalmemory.retrieval.query_embedding import QueryEmbedder

    class _Embedder:
        is_warm = True
        has_loaded_once = True

        def embed(self, text):
            return [0.1, 0.2]

    qe = QueryEmbedder(lambda: _Embedder())
    with _recall_in_flight() as step:
        assert recall_gate.recalls_needing_embedder() == 1
        out: list = []
        step(lambda: out.append(qe.embed("which port?", 1.0)))
        assert out == [([0.1, 0.2], None)]
        assert recall_gate.recalls_needing_embedder() == 0
        assert recall_gate.in_flight() == 1
