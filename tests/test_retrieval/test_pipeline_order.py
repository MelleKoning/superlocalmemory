# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Retrieval pipeline ordering and timeout hygiene tests.

Tests in this file pin two observable contracts:

1. The evidence floor must filter candidates BEFORE the cross-encoder reranker
   sees them (reducing the CE batch to the already-qualified pool).

2. Channel futures that exceed the latency budget must have cancel() called
   on them (preventing orphaned entries from accumulating in the thread pool queue).
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.storage.models import AtomicFact, Mode

# Every RetrievalEngine a test builds is closed when the test ends.
pytestmark = pytest.mark.usefixtures("closes_retrieval_engines")


# ---------------------------------------------------------------------------
# Shared helpers (mirrors pattern from test_evidence_floor.py)
# ---------------------------------------------------------------------------

def _make_fact(fact_id: str, content: str = "") -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content=content or f"content for {fact_id}",
        confidence=0.9,
    )


def _mock_db(facts: list[AtomicFact]) -> MagicMock:
    db = MagicMock()
    db.get_facts_by_ids.side_effect = lambda ids, pid, **kw: [
        f for f in facts if f.fact_id in ids
    ]
    db.get_scenes_for_facts_batch.return_value = {}
    # Correction-admission contract — real empty sets prove no corrections active.
    db.get_invalidated_fact_ids.return_value = set()
    db.get_nonapplied_correction_successor_ids.return_value = set()
    db.get_strict_temporal_excluded_fact_ids.return_value = set()
    return db


class _SpyReranker:
    """Cross-encoder spy: records what candidates it receives, returns them unchanged.

    Deliberately has NO rerank_with_status attribute so the engine falls
    through to the simpler rerank() contract (avoiding MagicMock auto-attrs).
    """

    _worker_ready = True
    # Per-instance storage; reset in tests.
    received_fact_ids: list[set[str]]

    def __init__(self) -> None:
        self.received_fact_ids = []

    def rerank(
        self,
        query: str,
        candidates: list[tuple[AtomicFact, float]],
        top_k: int | None = None,
    ) -> list[tuple[AtomicFact, float]]:
        self.received_fact_ids.append({fact.fact_id for fact, _ in candidates})
        return candidates


# ---------------------------------------------------------------------------
# 1.4 — Evidence floor runs before the cross-encoder
# ---------------------------------------------------------------------------

class TestFloorBeforeReranker:
    """The evidence floor must filter candidates before the cross-encoder."""

    def _build_mixed_engine(
        self, reranker: _SpyReranker,
    ) -> RetrievalEngine:
        """Engine with one floor-passing fact (f_pass, bm25>0) and one floor-failing
        fact (f_fail, zero channel evidence).

        f_fail has semantic=0.30 (below 0.60 default) and no bm25/temporal/entity,
        so it should be removed by the floor before the reranker sees it.
        """
        f_pass = _make_fact("f_pass", "microservices architecture decision")
        f_fail = _make_fact("f_fail", "associative noise candidate")
        db = _mock_db([f_pass, f_fail])

        sem = MagicMock()
        sem.search.return_value = [
            ("f_pass", 0.30),  # below 0.60 floor, but passes via bm25
            ("f_fail", 0.30),  # below 0.60 floor, no other channel evidence
        ]
        bm = MagicMock()
        bm.search.return_value = [("f_pass", 1.5)]  # f_pass: bm25 > 0 → passes floor
        emb = MagicMock()
        emb.embed.return_value = [0.1, 0.2, 0.3]

        return RetrievalEngine(
            db=db,
            config=RetrievalConfig(),
            channels={"semantic": sem, "bm25": bm},
            embedder=emb,
            reranker=reranker,
        )

    def test_reranker_receives_only_floor_qualified_candidates(self) -> None:
        """The cross-encoder must only receive candidates that passed the evidence floor.

        A zero-evidence candidate must be removed by the floor before the
        cross-encoder scores it.
        """
        spy = _SpyReranker()
        engine = self._build_mixed_engine(spy)
        engine.recall("microservices", "default", Mode.A, limit=10)

        assert spy.received_fact_ids, "Reranker must have been called"
        received = spy.received_fact_ids[0]

        assert "f_fail" not in received, (
            "f_fail has no channel evidence (semantic 0.30, bm25=0); "
            "evidence floor must remove it BEFORE the cross-encoder sees it"
        )
        assert "f_pass" in received, (
            "f_pass has bm25=1.5 evidence; it must reach the cross-encoder"
        )

    def test_nonsense_query_still_abstains_after_reordering(self) -> None:
        """Abstention behaviour must be stable regardless of pipeline order.

        A query that earns zero channel evidence on ALL facts must still
        return no results and no_confident_match=True.
        """
        # All facts have zero channel evidence → full abstention expected
        f1 = _make_fact("f1", "completely unrelated content")
        db = _mock_db([f1])

        sem = MagicMock()
        sem.search.return_value = [("f1", 0.10)]  # well below floor
        bm = MagicMock()
        bm.search.return_value = []  # no bm25 hit
        emb = MagicMock()
        emb.embed.return_value = [0.1, 0.2, 0.3]

        spy = _SpyReranker()
        engine = RetrievalEngine(
            db=db,
            config=RetrievalConfig(),
            channels={"semantic": sem, "bm25": bm},
            embedder=emb,
            reranker=spy,
        )

        response = engine.recall(
            "purple elephant quantum knitting", "default", Mode.A, limit=10,
        )

        assert response.results == [], "Nonsense query must return no results"
        assert response.no_confident_match is True, (
            "no_confident_match must be True when all candidates fail the floor"
        )

    def test_floor_qualified_candidate_appears_in_output(self) -> None:
        """A candidate that passes the floor must appear in the final results.

        Regression guard: the pipeline reordering must not silently drop
        floor-qualified candidates.
        """
        f1 = _make_fact("f1", "Python asyncio event loop internals")
        db = _mock_db([f1])

        sem = MagicMock()
        sem.search.return_value = [("f1", 0.75)]  # above 0.60 floor
        bm = MagicMock()
        bm.search.return_value = []
        emb = MagicMock()
        emb.embed.return_value = [0.1, 0.2, 0.3]

        engine = RetrievalEngine(
            db=db,
            config=RetrievalConfig(),
            channels={"semantic": sem, "bm25": bm},
            embedder=emb,
        )

        response = engine.recall("asyncio", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, (
            "f1 passed the evidence floor (semantic=0.75); must appear in results"
        )


# ---------------------------------------------------------------------------
# 1.5 — Timed-out channel futures are cancelled
# ---------------------------------------------------------------------------

class TestTimeoutFutureCancellation:
    """Channel futures exceeding the latency budget must be cancelled."""

    def _build_single_channel_engine(self) -> RetrievalEngine:
        """Engine with bm25 only; simpler future accounting."""
        f1 = _make_fact("f1", "some content")
        db = _mock_db([f1])
        bm = MagicMock()
        bm.search.return_value = [("f1", 1.0)]
        emb = MagicMock()
        emb.embed.return_value = [0.1, 0.2, 0.3]
        return RetrievalEngine(
            db=db,
            config=RetrievalConfig(),
            channels={"bm25": bm},
            embedder=emb,
        )

    def test_pending_futures_have_cancel_called(self) -> None:
        """A future in the pending (timed-out) set must have cancel() called on it.

        cancel() on an already-running thread is a no-op, but it is required
        for futures still in the queue and marks the future's state correctly.
        """
        engine = self._build_single_channel_engine()

        mock_future: MagicMock = MagicMock(spec=concurrent.futures.Future)
        # Make the executor return our controlled mock future
        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        engine._channel_executor = mock_executor

        # Simulate a timeout: the future lands in the pending set
        with patch(
            "superlocalmemory.retrieval.engine.concurrent.futures.wait",
            return_value=(set(), {mock_future}),
        ):
            engine.recall("any query", "default", Mode.A, limit=10)

        mock_future.cancel.assert_called(), (
            "cancel() must be called on a future in the pending (timed-out) set"
        )

    def test_non_pending_futures_are_not_cancelled(self) -> None:
        """Futures that completed on time must NOT have cancel() called.

        This ensures the cancel() call is scoped only to the pending set and
        does not accidentally cancel completed work.
        """
        engine = self._build_single_channel_engine()

        mock_done_future: MagicMock = MagicMock(spec=concurrent.futures.Future)
        # Done future returns a valid channel result
        mock_done_future.result.return_value = ("bm25", [("f1", 1.0)])
        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_done_future
        engine._channel_executor = mock_executor

        # The future completes on time: it is in done, not pending
        with patch(
            "superlocalmemory.retrieval.engine.concurrent.futures.wait",
            return_value=({mock_done_future}, set()),  # pending is empty
        ):
            engine.recall("any query", "default", Mode.A, limit=10)

        mock_done_future.cancel.assert_not_called(), (
            "cancel() must NOT be called on a future that completed on time"
        )
