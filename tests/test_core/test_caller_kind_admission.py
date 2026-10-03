# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A kind declared when a memory is saved is confirmed, on every fact, from the
moment it is searchable through enrichment; anything else never breaks a save."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingestion_command import IngestionRequest, IngestionState
from superlocalmemory.core.kind_assignment import assign_kinds
from superlocalmemory.storage.memory_kinds import METADATA_KEY, KindAssignment, KindSource, MemoryKind
from superlocalmemory.storage.migrations import M018_ingestion_operations
from superlocalmemory.storage.models import AtomicFact, FactType

CONTENT = ("We decided to ship the answer check in 4.1.18 and keep reordering "
           "behind its own switch for every user.")


def _command(engine):
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    return build_engine_ingestion_command(engine)


def _request(engine, *, kind=None, source_type="http", key="op-kind-1") -> IngestionRequest:
    metadata = {} if kind is None else {METADATA_KEY: kind}
    return IngestionRequest(
        content=CONTENT, profile_id=engine._profile_id, source_type=source_type,
        idempotency_key=key, metadata=metadata,
        trusted_actor_id="daemon-capability:owned-instance",
    )


def _row(engine, fact_id: str) -> dict:
    return dict(engine._db.execute(
        "SELECT fact_type, memory_kind, memory_kind_source FROM atomic_facts WHERE fact_id=?",
        (fact_id,))[0])


def test_kind_on_remember_is_confirmed_on_the_queryable_fact(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    receipt = _command(engine).submit(_request(engine, kind="decision"))
    assert receipt.state is IngestionState.QUERYABLE
    row = _row(engine, receipt.queryable_fact_ids[0])
    assert row == {"fact_type": "episodic", "memory_kind": "decision",
                   "memory_kind_source": "caller"}


def test_bad_kind_is_stored_untyped(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    receipt = _command(engine).submit(_request(engine, kind="DROP TABLE x"))
    assert receipt.state is IngestionState.QUERYABLE
    assert _row(engine, receipt.queryable_fact_ids[0])["memory_kind"] is None


def test_an_automatic_capture_cannot_confirm_a_kind(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    receipt = _command(engine).submit(_request(engine, kind="rule", source_type="http-observe"))
    assert _row(engine, receipt.queryable_fact_ids[0])["memory_kind"] is None


def test_replay_of_a_4_1_18_journal_entry_without_kind(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    receipt = _command(engine).submit(_request(engine))
    row = _row(engine, receipt.queryable_fact_ids[0])
    assert row["memory_kind"] is None and row["fact_type"] == "episodic"


def test_caller_kind_survives_materialization(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    command = _command(engine)
    receipt = command.submit(_request(engine, kind="rule"))
    derived = [
        AtomicFact(fact_id="derived-1", content="Ship the answer check in 4.1.18.",
                   fact_type=FactType.EPISODIC, confidence=0.9),
        AtomicFact(fact_id="derived-2", content="Reordering stays behind its own switch.",
                   fact_type=FactType.OPINION, confidence=0.9),
    ]
    with patch.object(engine._fact_extractor, "extract_facts", return_value=derived):
        done = command.materialize(receipt.operation_id)
    assert done.state is IngestionState.COMPLETE
    assert len(done.final_fact_ids) >= 2
    for fact_id in done.final_fact_ids:
        row = _row(engine, fact_id)
        assert row == {"fact_type": "semantic", "memory_kind": "rule",
                       "memory_kind_source": "caller"}, (fact_id, row)


class _Db:
    def __init__(self, has: bool) -> None:
        self._has = has

    def has_memory_kind_columns(self) -> bool:
        return self._has


class _Suggests:
    def __init__(self, kind=MemoryKind.STATUS, raises=False) -> None:
        self._kind, self._raises = kind, raises

    def suggest(self, facts, *, caller_kind):
        if self._raises:
            raise RuntimeError("model went away")
        return [KindAssignment(self._kind, KindSource.MODEL_LAYA, 0.8, "kinds-v1")] * len(facts)


def _fact() -> AtomicFact:
    return AtomicFact(fact_id="f", content="The build is green now.", fact_type=FactType.SEMANTIC)


def test_suggestions_never_change_fact_type() -> None:
    out = assign_kinds([_fact()], metadata={}, source_type="http", classifier=_Suggests(),
                       db=_Db(True))
    assert out[0].memory_kind == "status" and out[0].memory_kind_source == "model:laya"
    assert out[0].fact_type is FactType.SEMANTIC


def test_a_failing_classifier_never_breaks_a_save() -> None:
    out = assign_kinds([_fact()], metadata={}, source_type="http",
                       classifier=_Suggests(raises=True), db=_Db(True))
    assert out[0].memory_kind is None


@pytest.mark.parametrize("metadata", [{}, {METADATA_KEY: "decision"}])
def test_pre_m052_store_pipeline_unchanged(metadata) -> None:
    fact = _fact()
    out = assign_kinds([fact], metadata=metadata, source_type="http", classifier=_Suggests(),
                       db=_Db(False))
    assert out[0] is fact


class _SuggestsNothing:
    """A classifier with no suggestion to give for any fact — the same shape
    ``KindClassifier._suggest`` returns when its backend resolves to OFF (it
    never returns a per-fact mix: either every fact gets ``None`` or none
    does)."""

    def suggest(self, facts, *, caller_kind):
        return [None] * len(facts)


def _fact_with_llm_hint(kind: str) -> AtomicFact:
    """A fact the Mode B/C extractor already tagged with a ``model:llm`` kind
    hint (``encoding.llm_kind_hint.with_hint``), exactly as it arrives at
    ``assign_kinds`` during real extraction — before this function decides
    whether a suggestion applies at all."""
    from superlocalmemory.encoding import llm_kind_hint

    return llm_kind_hint.with_hint(_fact(), {"kind": kind})


def test_no_suggestion_clears_a_hint_the_fact_already_carried() -> None:
    """L2-12: memory kinds OFF (or any other no-suggestion outcome) must
    store no machine kind at all — not merely withhold a new one. An
    extractor-attached ``model:llm`` hint predates the OFF/no-suggestion
    decision and must be blanked, the same clearing
    ``encoding.memory_kind_classifier.with_kind`` already documents for its
    own backend-off case. A caller-declared kind is unaffected: that path
    never produces ``assignment is None`` (``_suggestions`` returns a CALLER
    assignment for every fact when one is declared)."""
    hinted = _fact_with_llm_hint("decision")
    assert hinted.memory_kind == "decision"  # sanity: the hint really landed

    out = assign_kinds([hinted], metadata={}, source_type="http",
                       classifier=_SuggestsNothing(), db=_Db(True))
    assert out[0].memory_kind is None
    assert out[0].memory_kind_source is None
    assert out[0].memory_kind_confidence is None
    assert out[0].memory_kind_recipe is None
    assert out[0].memory_kind_at is None
    assert out[0].fact_type is FactType.SEMANTIC  # a suggestion never touches it


def test_no_classifier_also_clears_a_hint_the_fact_already_carried() -> None:
    hinted = _fact_with_llm_hint("status")
    out = assign_kinds([hinted], metadata={}, source_type="http", classifier=None,
                       db=_Db(True))
    assert out[0].memory_kind is None
    assert out[0].memory_kind_source is None
