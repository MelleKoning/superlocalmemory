# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every recall carries how long its answer check took — and the envelope does not."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from superlocalmemory.core import answer_check_history as h
from superlocalmemory.core import judge_selection, recall_pipeline
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.server.recall_serializer import recall_response_metadata
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

#: The recall envelope's keys as of 4.1.19. The trace must never appear here:
#: MCP and HTTP output stay byte-identical.
_ENVELOPE_4_1_19 = {
    "score_contract_version", "calibration_status", "calibration_id", "query_id",
    "answer_confidence", "abstained", "abstention_reason", "temporal_frame",
    "thematic_context", "incomplete_channels", "channel_status", "reranker_status",
    "local_reranker_status", "answer_check_status",
}


@pytest.fixture(autouse=True)
def _no_live_judge(monkeypatch):
    monkeypatch.setattr(judge_selection, "_live", None)


class _SlowJudge:
    backend = "laya"
    top_k = 3

    def assess(self, query, documents, *, deadline=None):
        time.sleep(0.05)
        return acs.JudgeOutcome(SufficiencyVerdict((0.2,), 0.5, "laya:test"),
                                acs.STATUS_JUDGED)

    def shutdown(self) -> None: ...


def _run(judge, config, monkeypatch, n: int = 5):
    response = RecallResponse(results=[
        RetrievalResult(fact=AtomicFact(fact_id=f"f{i}", content=f"memory {i}",
                                        confidence=0.8), score=0.9, confidence=1.0)
        for i in range(n)])
    engine = SimpleNamespace(_sufficiency_judge=judge, recall=lambda *a, **k: response)
    monkeypatch.setattr(recall_pipeline, "apply_ranking", lambda resp, *a, **k: resp)
    return recall_pipeline.run_recall(
        "which one?", "default", fast=True, config=config, retrieval_engine=engine,
        trust_scorer=None, embedder=None, db=SimpleNamespace(db_path=None), llm=None,
        hooks=None)


def test_judge_time_is_measured(mode_a_config, monkeypatch) -> None:
    out = _run(_SlowJudge(), mode_a_config, monkeypatch)
    trace = out.answer_check_trace
    assert isinstance(trace, acs.AnswerCheckTrace)
    assert trace.judge_ms >= 50.0
    assert trace.total_ms >= trace.retrieval_ms + trace.judge_ms - 0.2  # rounding
    assert trace.backend == "laya" and trace.threshold == 0.5 and trace.detail == ""
    assert out.abstained is True and out.answer_check_status == acs.STATUS_JUDGED


def test_no_judge_means_no_backend_and_no_wait(mode_a_config, monkeypatch) -> None:
    out = _run(None, mode_a_config, monkeypatch)
    trace = out.answer_check_trace
    assert trace.backend == "" and trace.judge_ms < 5.0
    assert out.answer_check_status == acs.STATUS_OFF


def test_no_results_is_explained(mode_a_config, monkeypatch) -> None:
    out = _run(_SlowJudge(), mode_a_config, monkeypatch, n=0)
    assert out.answer_check_trace.detail == acs.DETAIL_NO_RESULTS
    assert out.answer_check_trace.judge_ms < 5.0


def test_envelope_unchanged_from_4_1_19(mode_a_config, monkeypatch) -> None:
    out = _run(_SlowJudge(), mode_a_config, monkeypatch)
    meta = recall_response_metadata(out)
    assert set(meta) == _ENVELOPE_4_1_19
    assert "answer_check_trace" not in repr(meta)


def test_engine_recall_records_once_with_the_answers_name(monkeypatch) -> None:
    """engine.recall is the one emission point; the event is named by query_id."""
    from superlocalmemory.core.engine import MemoryEngine

    h._reset_for_testing()
    h.enable(True)
    try:
        resp = RecallResponse(results=[])
        resp.answer_check_status = acs.STATUS_OFF
        engine = MemoryEngine.__new__(MemoryEngine)
        engine._config = SimpleNamespace(scope=None)
        engine._profile_id = "default"
        monkeypatch.setattr(MemoryEngine, "_require_full", lambda self, name: None)
        monkeypatch.setattr(MemoryEngine, "_ensure_init", lambda self: None)
        monkeypatch.setattr(MemoryEngine, "_session_for_signals", lambda self, s: "s")
        for name in ("_retrieval_engine", "_trust_scorer", "_embedder", "_db", "_llm",
                     "_hooks", "_access_log", "_auto_linker"):
            setattr(engine, name, None)
        monkeypatch.setattr(recall_pipeline, "run_recall", lambda *a, **k: resp)
        out = engine.recall("q")
        items, _ = h.recent("default", after_seq=0, limit=10)
        assert len(items) == 1 and items[0][1].event_id == out.query_id
    finally:
        h._reset_for_testing()


def test_details_on_each_skip_branch() -> None:
    from superlocalmemory.core.answer_check_scope import skip_answer_check
    from superlocalmemory.core.answer_check_stage import run_answer_check

    judge = _SlowJudge()
    engine = SimpleNamespace(_sufficiency_judge=judge)
    full = RecallResponse(results=[RetrievalResult(
        fact=AtomicFact(fact_id="f", content="m", confidence=0.8), score=0.9, confidence=1.0)])
    with skip_answer_check():
        assert run_answer_check(engine, "q", full).detail == acs.DETAIL_NOT_A_QUESTION
    assert run_answer_check(engine, "q", RecallResponse()).detail == acs.DETAIL_NO_RESULTS
    out = run_answer_check(engine, "q", full, recall_started=time.monotonic() - 10)
    assert out.detail == acs.DETAIL_BUDGET and out.status == acs.STATUS_SKIPPED
    shared = RecallResponse(results=[RetrievalResult(
        fact=AtomicFact(fact_id="f", content="m", confidence=0.8, profile_id="other"),
        score=0.9, confidence=1.0)])
    jev = SimpleNamespace(backend="jev", top_k=3, assess=judge.assess)
    out = run_answer_check(SimpleNamespace(_sufficiency_judge=jev), "q", shared,
                           profile_id="default")
    assert out.detail == acs.DETAIL_OTHER_PROFILE
    assert run_answer_check(SimpleNamespace(_sufficiency_judge=None), "q", full).detail == ""


def test_detail_does_not_change_equality() -> None:
    assert (acs.JudgeOutcome(None, acs.STATUS_SKIPPED, "budget")
            == acs.JudgeOutcome(None, acs.STATUS_SKIPPED))
