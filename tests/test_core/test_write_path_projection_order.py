# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A fact is reported findable by meaning only when it actually is.

Two representations are involved and they answer different questions. The vector
projection is what a search on meaning queries, so it alone decides whether a
fact can be *found*. The canonical ``atomic_facts.embedding`` column is where the
vector *lives* — what an export, a backup or a direct fetch reads.

The rules that follow from that, and each one here is a defect that shipped:

* The projection is attempted first, because that ordering keeps the two
  consistent whenever both can succeed.
* The column is written either way. The vector is real data that has already been
  computed, and every reason a projection refuses — no search extension on this
  platform, a store at a different dimension — is a property of the installation
  rather than of one fact, so withholding one fact's column repairs nothing and
  discards the vector.
* Whether a fact is "findable by meaning" is read off the projection, never off
  the column, and never claimed when the projection did not accept the vector.
  That claim, not the column, is what misleads a caller into believing a memory
  is retrievable when no search can reach it.
* If the projection succeeded and the column write then failed, the projection is
  rolled back, so a search cannot return a fact whose vector nothing else holds.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import Mode


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[MemoryEngine]:
    eng = MemoryEngine(
        SLMConfig(
            mode=Mode.A,
            base_dir=tmp_path,
            db_path=tmp_path / "memory.db",
            active_profile="default",
            embedding=EmbeddingConfig(model_name="nomic-embed-text", dimension=768),
        )
    )
    eng._ensure_init()
    yield eng
    eng.close()


def _local_embedder(vector: list[float] | None = None) -> MagicMock:
    """A mock the engine classifies as a LOCAL, available embedder.

    ``_is_remote_embedder`` reads ``embedder._config``, and a bare ``MagicMock``
    auto-creates that attribute, so ``is_cloud`` comes back as a truthy mock and
    the engine returns early having done nothing. ``_config = None`` is what a
    local embedder looks like; without it a test here passes while covering
    nothing.
    """
    m = MagicMock()
    m._config = None
    m._available = True
    m.embed.return_value = vector if vector is not None else [0.01] * 768
    m.compute_fisher_params.return_value = (0.0, 1.0)
    return m


def _column_kind(engine: MemoryEngine, fact_id: str) -> str:
    """``typeof`` of the canonical column: 'null', 'blob', or 'text'."""
    conn = sqlite3.connect(f"file:{engine._config.db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT typeof(embedding) FROM atomic_facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        return row[0] if row else "missing"
    finally:
        conn.close()


class _Projection:
    """Stands in for the vector projection and records what was asked of it."""

    def __init__(self, *, accepts: bool = True, raises: bool = False) -> None:
        self.available = True
        self._accepts = accepts
        self._raises = raises
        self.held: set[str] = set()
        self.upserts: list[str] = []
        self.deletes: list[str] = []

    def upsert(self, fact_id: str, profile_id: str, embedding, model_name: str = "") -> bool:
        self.upserts.append(fact_id)
        if self._raises:
            raise RuntimeError("vec0 extension not loaded")
        if not self._accepts:
            return False
        self.held.add(fact_id)
        return True

    def delete(self, fact_id: str) -> bool:
        self.deletes.append(fact_id)
        self.held.discard(fact_id)
        return True

    def raw_vector_present(self, fact_id: str) -> bool:
        return fact_id in self.held

    def is_searchable_by_meaning(self, fact_id: str, profile_id: str | None = None) -> bool:
        """The question the engine actually asks.

        Without this method the engine's check fails closed and every test in
        this file takes the re-project path — so the "already findable" branch
        was never reached, and an implementation of it that always answered
        "no" would have passed.
        """
        return fact_id in self.held


class TestARefusedProjectionIsNeverReportedAsFindable:
    """The case that makes a memory silently unfindable if it is got wrong.

    A projection can refuse for reasons that have nothing to do with the fact in
    hand: no search extension on this platform, a store left at a different
    dimension after a mode change. Every one of those is a property of the
    installation, so it applies to every fact equally.

    That is why the vector is still written to the canonical column — it is real
    data, already computed, and the only place it can live — and why the *report*
    is the thing that must not lie. An earlier version of this file asserted the
    opposite, that a refused projection should leave the column NULL so a repair
    pass keyed on that column would retry. It would not help: the same
    installation-wide reason would refuse the retry too, so the only effect was
    to discard the vector and pay for the model again on every pass.
    """

    def test_the_vector_is_still_stored_when_the_projection_is_refused(
        self, engine: MemoryEngine
    ) -> None:
        engine._embedder = _local_embedder()
        engine._vector_store = _Projection(accepts=False)

        fact_id = engine.store_fast("The Helsinki ledger reconciliation is deferred.")[0]

        assert engine._vector_store.upserts == [fact_id], (
            "the projection was never attempted, so this test is not exercising "
            "the path it claims to"
        )
        assert _column_kind(engine, fact_id) == "blob", (
            "the vector was computed and then thrown away; the column is the only "
            "place it can live when the projection will not take it"
        )

    def test_a_refused_projection_is_not_counted_as_findable(
        self, engine: MemoryEngine
    ) -> None:
        """The assertion that protects the user.

        The count is what a receipt turns into "findable by meaning". Saying that
        about a fact no search can reach is the actual harm — not the column.
        """
        engine._embedder = _local_embedder()
        engine._vector_store = _Projection(accepts=False)
        fact_id = engine.store_fast("Procurement confirmed the tariff schedule.")[0]

        assert engine.enrich_new_facts_now([fact_id]) == 0, (
            "a fact the projection refused was counted as findable by meaning, so "
            "the receipt claims a search can reach a memory that it cannot"
        )

    def test_a_projection_that_raises_reaches_the_same_end_state(
        self, engine: MemoryEngine
    ) -> None:
        """Raising and refusing must be indistinguishable to the caller.

        The previous code swallowed the exception with a bare ``except: pass``
        after already reporting success, so a raising projection produced a fact
        the receipt called searchable and no search could return.
        """
        engine._embedder = _local_embedder()
        engine._vector_store = _Projection(raises=True)

        fact_id = engine.store_fast("The Bergen invoice batch was rejected.")[0]

        assert _column_kind(engine, fact_id) == "blob"
        assert engine.enrich_new_facts_now([fact_id]) == 0

    def test_a_successful_projection_is_reported_as_findable(
        self, engine: MemoryEngine
    ) -> None:
        """The happy path, without which every test above passes vacuously.

        An implementation that always refused would satisfy all three.
        """
        engine._embedder = _local_embedder()
        engine._vector_store = _Projection(accepts=True)

        fact_id = engine.store_fast("The Oslo migration window opens on Friday.")[0]

        assert engine._vector_store.raw_vector_present(fact_id)
        assert _column_kind(engine, fact_id) == "blob"
        assert engine.enrich_new_facts_now([fact_id]) == 1


class TestFindableIsDecidedByTheProjection:
    """"It already has a vector" must be read off the projection, not the column."""

    def test_a_column_without_a_projection_is_not_counted_as_findable(
        self, engine: MemoryEngine
    ) -> None:
        """This is the state a half-completed write leaves behind.

        Counting such a fact as findable makes the receipt claim "searchable by
        meaning" about a fact the semantic channel cannot reach, and skips the
        one chance to repair it.
        """
        engine._embedder = _local_embedder()
        engine._vector_store = _Projection(accepts=True)
        fact_id = engine.store_fast("The Lisbon rollout is paused.")[0]

        # Simulate the damaged state: column written, projection missing.
        engine._vector_store.held.discard(fact_id)
        engine._vector_store.upserts.clear()

        count = engine.enrich_new_facts_now([fact_id])

        assert engine._vector_store.upserts == [fact_id], (
            "the fact was passed over because its column was populated; the "
            "column is not what the semantic channel searches, so it was "
            "reported findable while remaining invisible"
        )
        assert count == 1
        assert engine._vector_store.raw_vector_present(fact_id)

    def test_repairing_from_the_column_does_not_pay_to_embed_again(
        self, engine: MemoryEngine
    ) -> None:
        """The vector is already on the row — reuse it rather than recompute it.

        Asserted separately because an implementation that re-embeds everything
        would satisfy the test above while making every repair cost a model call.
        """
        engine._embedder = _local_embedder()
        engine._vector_store = _Projection(accepts=True)
        fact_id = engine.store_fast("The Bergen invoice batch was rejected.")[0]
        engine._vector_store.held.discard(fact_id)

        engine._embedder.embed.reset_mock()
        engine.enrich_new_facts_now([fact_id])

        engine._embedder.embed.assert_not_called()


class TestTheTwoWritesDoNotDriftApart:
    def test_a_failed_column_write_rolls_the_projection_back(
        self, engine: MemoryEngine
    ) -> None:
        """The opposite split-brain: projection present, column empty.

        The semantic channel would find this fact while anything reading the
        column — a direct fetch, an export, a backup — sees no vector at all.
        Rolling the projection back puts the row into the one state that heals
        itself: NULL column, no projection, picked up by the repair pass.
        """
        engine._embedder = _local_embedder()
        engine._vector_store = _Projection(accepts=True)

        fact_id = engine.store_fast("The Turku contract renews in March.")[0]
        engine._vector_store.held.discard(fact_id)

        engine._db.update_fact = MagicMock(side_effect=sqlite3.OperationalError("database is locked"))
        engine.enrich_new_facts_now([fact_id])

        assert engine._vector_store.deletes == [fact_id], (
            "the canonical write failed and the projection was left behind, so "
            "the two representations now disagree with nothing to reconcile them"
        )
        assert not engine._vector_store.raw_vector_present(fact_id)


class _CloseOnAcquire:
    """A lock that completes a shutdown the first time it is acquired.

    Reproduces the one interleaving that matters and that no sequential call can
    reach: a caller passes the unlocked "there is no pool yet" check, shutdown
    runs to completion, and only then does the caller arrive at the guarded
    section. Reentrant, because the shutdown path takes this same lock.
    """

    def __init__(self, engine: MemoryEngine) -> None:
        import threading

        self._engine = engine
        self._inner = threading.RLock()
        self._fired = False

    def __enter__(self) -> bool:
        self._inner.acquire()
        if not self._fired:
            self._fired = True
            self._engine.close()
        return True

    def __exit__(self, *_exc: object) -> bool:
        self._inner.release()
        return False


class TestTheEmbedPoolStaysClosed:
    def test_a_call_that_races_shutdown_does_not_resurrect_the_pool(
        self, engine: MemoryEngine
    ) -> None:
        """Shutdown has to be final, or it is not shutdown.

        The check for "is there a pool yet" is deliberately unlocked, so a caller
        can pass it, lose the CPU, and arrive at the guarded section after the
        engine has already been closed. Finding no pool there, it would build a
        replacement that nothing owns and nothing will ever shut down — so a
        burst of writes arriving during shutdown leaks a thread each and holds
        the interpreter open past its shutdown budget, which is the failure the
        shutdown path exists to prevent.

        Sequential calls after close() are already covered incidentally, because
        closing also drops the embedder. That is not this defect, and a test
        built that way passes without the guard.
        """
        engine._embedder = _local_embedder()
        engine._store_fast_embed_pool = None
        engine._store_fast_embed_pool_lock = _CloseOnAcquire(engine)

        emb, _fmean, _fvar = engine._warm_guard_embed("A write racing shutdown.")

        assert engine._store_fast_embed_pool is None, (
            "a new embed pool was created after close(); it has no owner and no "
            "shutdown, so it keeps the interpreter alive"
        )
        assert emb is None, "the racing call must decline rather than embed"
