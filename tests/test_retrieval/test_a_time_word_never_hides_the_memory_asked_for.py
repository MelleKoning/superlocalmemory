# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A time word in a question narrows recall only when the narrowed set still answers.

Recall reads a scope like "recently" or "last week" from the question and keeps
only the candidates dated inside it. That inferred window was meant to be
additive ("never makes recall worse"), but it fell back only when it kept
nothing at all. Measured on a copy of a 22k-fact store: the first 12 or 40
words of a stored memory, given back as the question, returned NO result at all
for 3 of 40 and 6 of 40 ordinary memories, and for 36 such questions in all
every one carried "recently"-type wording (window 30 days) about a memory
written in August. The window kept a few unrelated recent facts with no
evidence, dropped the memory itself, and the evidence floor then removed the
rest.
"""

from __future__ import annotations

import inspect

from superlocalmemory.retrieval.fusion import FusionResult
from superlocalmemory.retrieval.time_window import windowed_candidates

MIN_SEM = 0.60


def _fr(fact_id: str, **scores: float) -> FusionResult:
    return FusionResult(fact_id=fact_id, fused_score=1.0, channel_ranks={},
                        channel_scores=dict(scores))


def test_an_inferred_window_with_no_evidence_inside_keeps_every_candidate() -> None:
    target = _fr("old-memory", bm25=7.0, semantic=0.91)          # written in August
    recent = _fr("recent-noise", hopfield=0.4, spreading_activation=0.2)
    fused = [target, recent]
    kept = windowed_candidates(fused, lambda fid: fid == "recent-noise",
                               explicit=False, min_semantic=MIN_SEM)
    assert [fr.fact_id for fr in kept] == ["old-memory", "recent-noise"]


def test_an_inferred_window_with_evidence_inside_still_narrows() -> None:
    old = _fr("old", bm25=2.0)
    new = _fr("new", semantic=0.72)
    kept = windowed_candidates([old, new], lambda fid: fid == "new",
                               explicit=False, min_semantic=MIN_SEM)
    assert [fr.fact_id for fr in kept] == ["new"]


def test_an_inferred_window_that_keeps_nothing_keeps_every_candidate() -> None:
    fused = [_fr("a", bm25=1.0), _fr("b", semantic=0.8)]
    kept = windowed_candidates(fused, lambda fid: False, explicit=False, min_semantic=MIN_SEM)
    assert [fr.fact_id for fr in kept] == ["a", "b"]


def test_an_explicit_window_is_honoured_even_when_it_answers_nothing() -> None:
    fused = [_fr("a", bm25=1.0), _fr("b", hopfield=0.9)]
    assert windowed_candidates(fused, lambda fid: False, explicit=True,
                               min_semantic=MIN_SEM) == []
    kept = windowed_candidates(fused, lambda fid: fid == "b", explicit=True,
                               min_semantic=MIN_SEM)
    assert [fr.fact_id for fr in kept] == ["b"]


def test_recall_uses_it() -> None:
    from superlocalmemory.retrieval import engine

    assert "windowed_candidates(" in inspect.getsource(engine.RetrievalEngine.recall)
