# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The reranker scores short candidates without padding them to the longest one.

One batch of 30-60 pairs was padded to its longest pair (a whole saved text,
512 tokens), so every short fact cost as much as the longest: 287 ms instead of
170 ms a recall on a live-store copy. Pairs are now scored shortest first in
small batches, and every score is returned in the caller's order.
"""

from __future__ import annotations

from superlocalmemory.core import reranker_worker as rw


class _Model:
    """Scores a pair by its document; records the batches it was given."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def predict(self, pairs, batch_size=32):
        out = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            self.batches.append([len(d) for _, d in chunk])
            out.extend(float(len(d)) + 0.5 for _, d in chunk)
        return out


def test_scores_come_back_in_the_callers_order() -> None:
    docs = ["x" * n for n in (300, 5, 120, 5, 40, 512, 7)]
    scores = rw.predict_by_length(_Model(), [("q", d) for d in docs], batch_size=3)
    assert scores == [float(len(d)) + 0.5 for d in docs]


def test_each_batch_holds_pairs_of_similar_length() -> None:
    model = _Model()
    docs = ["x" * n for n in (512, 10, 500, 12, 11, 490, 9, 13)]
    rw.predict_by_length(model, [("q", d) for d in docs], batch_size=4)
    assert model.batches == [[9, 10, 11, 12], [13, 490, 500, 512]]


def test_no_pairs_no_call() -> None:
    model = _Model()
    assert rw.predict_by_length(model, []) == [] and model.batches == []


def test_a_scorer_without_batch_size_still_answers_in_order() -> None:
    class Plain:
        def predict(self, pairs):
            return [float(len(d)) for _, d in pairs]

    docs = ["x" * n for n in (30, 2, 11)]
    assert rw.predict_by_length(Plain(), [("q", d) for d in docs]) == [30.0, 2.0, 11.0]
