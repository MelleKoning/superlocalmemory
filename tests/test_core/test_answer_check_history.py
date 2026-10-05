# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The in-memory side of the Answer Check history: O(1), no I/O, no text."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import os
import sqlite3
import statistics
import time
import uuid
from pathlib import Path

import pytest

from superlocalmemory.core import answer_check_history as h
from superlocalmemory.core.answer_check_scope import skip_answer_check
from superlocalmemory.retrieval.answer_check_status import AnswerCheckTrace
from superlocalmemory.storage.models import RecallResponse

_HOT = Path(h.__file__)


def make_response(*, status="judged", abstained=False, confidence=0.8, results=3,
                  query_id=None, trace=None, reason=None, query_type="factual",
                  calibration_id="laya-v1:0.5") -> RecallResponse:
    resp = RecallResponse(query="SECRET-Q", results=[object()] * results)
    resp.answer_check_status = status
    resp.abstained = abstained
    resp.abstention_reason = reason
    resp.answer_confidence = confidence
    resp.query_type = query_type
    resp.calibration_id = calibration_id
    resp.query_id = query_id if query_id is not None else uuid.uuid4().hex
    resp.answer_check_trace = trace or AnswerCheckTrace(
        detail="", backend="laya", threshold=0.5, reordered=False,
        retrieval_ms=700.0, judge_ms=200.0, total_ms=910.0)
    return resp


@pytest.fixture(autouse=True)
def fresh():
    h._reset_for_testing()
    yield
    h._reset_for_testing()


def test_record_is_noop_without_writer() -> None:
    for _ in range(100):
        h.record_recall_verdict(make_response(), profile_id="default")
    assert h.recent("default", after_seq=0, limit=50)[0] == []
    assert all(v == 0 for v in h.counters().values())


def test_event_has_no_text_fields() -> None:
    assert {f.name for f in dataclasses.fields(h.VerdictEvent)} == {
        "event_id", "profile_id", "occurred_ms", "status", "detail", "backend", "origin",
        "abstained", "abstention_reason", "answer_confidence", "threshold", "reordered",
        "result_count", "query_type", "retrieval_ms", "judge_ms", "total_ms",
        "calibration_id", "embed_ms", "rerank_ms"}


def test_event_carries_nothing_of_the_query_or_memories() -> None:
    ev = h.event_from_response(make_response(), "default", now_ms=1, origin_name="")
    assert "SECRET-Q" not in repr(ev)


def test_event_from_response_whitelists_enums() -> None:
    trace = AnswerCheckTrace(detail="x", backend="gpt", threshold=float("inf"),
                             reordered="yes", retrieval_ms=float("nan"), judge_ms=-5.0,
                             total_ms=10**9)
    resp = make_response(status="weird", confidence=1.7, trace=trace, query_type="Fact s",
                         reason="made_up", query_id="not-hex", calibration_id="a b")
    ev = h.event_from_response(resp, "default", now_ms=5, origin_name="evil")
    assert ev.status == "unavailable" and ev.detail == "" and ev.backend == ""
    assert ev.origin == "" and ev.query_type == "" and ev.abstention_reason is None
    assert ev.answer_confidence == 1.0 and ev.threshold is None
    assert ev.retrieval_ms is None and ev.judge_ms == 0.0 and ev.total_ms == 600_000.0
    assert ev.reordered is False and ev.calibration_id == ""
    assert len(ev.event_id) == 32 and ev.event_id != "not-hex"
    assert h.event_from_response(resp, "", now_ms=5, origin_name="") is None


def test_not_a_question_is_counted_not_recorded() -> None:
    h.enable(True)
    with skip_answer_check():
        h.record_recall_verdict(make_response(), profile_id="default")
        h.record_recall_verdict(make_response(status="off"), profile_id="default")
    assert h.recent("default", after_seq=0, limit=50)[0] == []
    assert h.counters()["not_a_question"] == 2
    assert h.counters()["recorded"] == 0


def test_full_ring_drops_oldest_and_counts() -> None:
    h.enable(True)
    for _ in range(h.RING_CAPACITY + 10):
        h.record_recall_verdict(make_response(), profile_id="default")
    assert h.counters()["dropped_before_save"] == 10
    assert h._ring[0][0] == 11
    assert h.counters()["recorded"] == h.RING_CAPACITY + 10


def test_inflight_eviction_not_double_counted() -> None:
    h.enable(True)
    for _ in range(128):
        h.record_recall_verdict(make_response(), profile_id="default")
    batch = h.snapshot_unsaved(128)
    for _ in range(h.RING_CAPACITY):
        h.record_recall_verdict(make_response(), profile_id="default")
    # The 128 in flight were evicted but are being saved: not counted as lost.
    assert h.counters()["dropped_before_save"] == 0
    h.mark_saved(batch[-1][0], len(batch))
    assert h.counters()["dropped_before_save"] == 0
    assert h.counters()["saved"] == 128


def test_mark_failed_counts_lost_inflight() -> None:
    h.enable(True)
    for _ in range(100):
        h.record_recall_verdict(make_response(), profile_id="default")
    batch = h.snapshot_unsaved(100)
    for _ in range(h.RING_CAPACITY - 60):   # 2,088 total: evicts the 40 oldest in flight
        h.record_recall_verdict(make_response(), profile_id="default")
    h.mark_failed(batch)
    c = h.counters()
    assert c["dropped_before_save"] == 40 and c["save_failures"] == 1
    assert len(h.snapshot_unsaved(10_000)) == h.RING_CAPACITY  # the rest are retried


def test_forget_profile_purges_ring_only_that_profile() -> None:
    h.enable(True)
    for _ in range(5):
        h.record_recall_verdict(make_response(), profile_id="a")
        h.record_recall_verdict(make_response(), profile_id="b")
    assert h.forget_profile("a") == 5
    assert h.recent("a", after_seq=0, limit=50)[0] == []
    assert len(h.recent("b", after_seq=0, limit=50)[0]) == 5
    assert h.counters()["erased_unsaved"] == 5


def test_forget_profile_with_cutoff_keeps_newer() -> None:
    h.enable(True)
    h.record_recall_verdict(make_response(), profile_id="a")
    cutoff = h._ring[-1][1].occurred_ms
    time.sleep(0.005)
    h.record_recall_verdict(make_response(), profile_id="a")
    assert h.forget_profile("a", occurred_before_ms=cutoff) == 1
    assert len(h.recent("a", after_seq=0, limit=50)[0]) == 1


def test_origin_contextvar_requires_known_name() -> None:
    h.enable(True)
    with pytest.raises(ValueError):
        with h.origin("x"):
            pass
    with h.origin("dashboard"):
        h.record_recall_verdict(make_response(), profile_id="default")
    h.call_as_dashboard(h.record_recall_verdict, make_response(), profile_id="default")
    h.record_recall_verdict(make_response(), profile_id="default")
    items, _ = h.recent("default", after_seq=0, limit=50)
    assert [ev.origin for _, ev in items] == ["", "dashboard", "dashboard"]


def test_view_origins_are_known_and_nothing_else_is() -> None:
    """4.1.21: a saved view run is tagged with where it ran; "" and made-up
    names are still refused."""
    h.enable(True)
    for name in ("", "view", "view-evil", "VIEW-CLI"):
        with pytest.raises(ValueError):
            with h.origin(name):
                pass
    for name in ("view-dashboard", "view-cli", "view-mcp"):
        with h.origin(name):
            h.record_recall_verdict(make_response(), profile_id="default")
    items, _ = h.recent("default", after_seq=0, limit=50)
    assert [ev.origin for _, ev in items] == ["view-mcp", "view-cli", "view-dashboard"]


def test_recent_is_newest_first_and_respects_after_seq() -> None:
    h.enable(True)
    for _ in range(5):
        h.record_recall_verdict(make_response(), profile_id="default")
    items, last = h.recent("default", after_seq=2, limit=50)
    assert [s for s, _ in items] == [5, 4, 3] and last == 5
    assert len(h.recent("default", after_seq=0, limit=2)[0]) == 2


_IO_MODULES = {"sqlite3", "os", "io", "pathlib", "socket", "urllib", "http", "shutil",
               "subprocess", "requests", "httpx"}


def test_hot_module_imports_no_io() -> None:
    tree = ast.parse(_HOT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & _IO_MODULES, imported & _IO_MODULES
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in calls


def test_record_does_no_io(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("I/O on the recall path")
    responses = [make_response() for _ in range(1000)]
    h.enable(True)
    monkeypatch.setattr(sqlite3, "connect", boom)
    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(os, "open", boom)
    for resp in responses:
        h.record_recall_verdict(resp, profile_id="default")
    assert h.counters()["recorded"] == 1000


def _median_ns(n: int) -> float:
    resp = make_response()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        h.record_recall_verdict(resp, profile_id="default")
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples)


def test_record_is_constant_time() -> None:
    h.enable(True)
    empty = _median_ns(500)
    for _ in range(h.RING_CAPACITY):
        h.record_recall_verdict(make_response(), profile_id="default")
    full = _median_ns(500)
    assert full < empty * 3, (empty, full)
