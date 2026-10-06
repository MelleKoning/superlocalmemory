# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every recall says whether its answer was checked: supported / unsupported / unjudged.

On a real store a question about something never stored came back with
candidates, the answer check skipped for lack of time, and ``abstained=false``
-- which every caller reads as "the memory answers this". It was never checked.
``answerability`` is the field that cannot be misread, derived the same way on
every surface, and only ``supported`` passes a gate that needs a checked answer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.core import answer_check_memo as memo
from superlocalmemory.core.answer_check_notice import answer_check_line
from superlocalmemory.core.answer_check_stage import run_answer_check
from superlocalmemory.mcp._recall_metadata import forward_recall_metadata
from superlocalmemory.mcp.tools_loops import register_loop_tools
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval.answerability import (
    REASONS,
    answerability,
    is_supported,
    of_envelope,
)
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.server.recall_serializer import recall_response_metadata
from superlocalmemory.storage.models import RecallResponse
from tests.mcp.test_loop_tools import _Capture, _FakeEngine


@pytest.mark.parametrize("status,detail,abstained,n,want", [
    ("judged", "", False, 3, ("supported", "judged_fresh")),
    ("judged", "reused", False, 3, ("supported", "judged_from_memo")),
    ("judged", "", True, 3, ("unsupported", "judged_fresh")),
    ("off", "", False, 3, ("unjudged", "disabled")),
    ("warming", "", False, 3, ("unjudged", "warming")),
    ("busy", "", False, 3, ("unjudged", "busy")),
    ("unavailable", "", False, 3, ("unjudged", "unavailable")),
    ("skipped", "budget", False, 3, ("unjudged", "budget_exhausted")),
    ("skipped", "not_a_question", False, 3, ("unjudged", "not_a_question")),
    ("skipped", "other_profile_memory", False, 3, ("unjudged", "other_profile")),
    ("skipped", "no_results", True, 0, ("unjudged", "no_results")),
    ("judged", "", False, 0, ("unjudged", "no_results")),   # nothing to have judged
    ("nonsense", "x", False, 3, ("unjudged", "unavailable")),
])
def test_every_check_outcome_maps_to_one_answerability(status, detail, abstained, n, want):
    got = answerability(status, detail, abstained=abstained, result_count=n)
    assert got == want and got[1] in REASONS


def test_the_fabricated_missing_answer_case_can_never_read_as_checked() -> None:
    # The M5 shape: candidates returned, check skipped on budget, abstained False.
    response = RecallResponse(query="q", results=[SimpleNamespace(fact=None)] * 3)
    response.answer_check_status, response.answer_check_detail = "skipped", "budget"
    response.abstained = False
    meta = recall_response_metadata(response)
    assert meta["abstained"] is False
    assert (meta["answerability"], meta["answerability_reason"]) == (
        "unjudged", "budget_exhausted")
    assert not is_supported(meta) and not is_supported(response)
    assert answer_check_line({**meta, "calibration_status": "shadow"}) == ""
    forwarded = forward_recall_metadata({**meta, "results": [{}, {}, {}]})
    assert forwarded["answerability"] == "unjudged"


def test_an_older_daemon_envelope_is_given_the_same_word() -> None:
    old = {"results": [{}], "abstained": False, "answer_check_status": "skipped",
           "answer_check_reason": "budget"}
    assert forward_recall_metadata(old)["answerability"] == "unjudged"
    assert of_envelope({"results": [{}], "abstained": False,
                        "answer_check_status": "judged"}) == ("supported", "judged_fresh")


# -- the bounded-loop gate --------------------------------------------------------

class _UnjudgedEngine(_FakeEngine):
    def recall(self, query, limit=3, fast=True, **kw):
        resp = super().recall(query, limit=limit, fast=fast)
        resp.answer_check_status = "skipped"
        resp.answer_check_detail = "budget"
        resp.abstained = False
        return resp


class _SupportedEngine(_UnjudgedEngine):
    def recall(self, query, limit=3, fast=True, **kw):
        resp = super().recall(query, limit=limit, fast=fast)
        resp.answer_check_status, resp.answer_check_detail = "judged", ""
        return resp


def _gate(engine_cls, **kw) -> dict:
    cap = _Capture()
    register_loop_tools(cap, engine_cls)
    return asyncio.run(cap.fns["slm_loop_run"](
        name="gate", gate_query="MATCH build passed", max_iterations=2,
        poll_interval_s=0.25, **kw))


def test_an_unchecked_match_never_passes_a_gate_that_needs_support() -> None:
    out = _gate(_UnjudgedEngine, require_support=True)
    assert out["ok"] is True and out["status"] != "DONE", out


def test_a_checked_sufficient_match_passes_it() -> None:
    assert _gate(_SupportedEngine, require_support=True)["status"] == "DONE"


def test_the_score_gate_alone_is_unchanged_by_default() -> None:
    assert _gate(_UnjudgedEngine)["status"] == "DONE"


# -- the reorder branch out of budget, and what a remembered verdict is bound to ---

class _Hosted:
    backend = "jev"
    rerank_enabled = True
    rerank_k = 5
    top_k = 3

    def __init__(self) -> None:
        self.asked = 0

    def rerank_and_judge(self, *a, **k):
        self.asked += 1
        raise AssertionError("asked out of budget")


def test_the_reorder_branch_out_of_budget_sends_nothing_and_keeps_the_order() -> None:
    import time

    judge = _Hosted()
    results = [SimpleNamespace(fact=SimpleNamespace(fact_id=f"f{i}", content=str(i),
                                                    profile_id="p")) for i in range(4)]
    response = SimpleNamespace(results=list(results))
    engine = SimpleNamespace(_sufficiency_judge=judge)
    out = run_answer_check(engine, "q", response, recall_started=time.monotonic() - 10.0,
                           profile_id="p")
    assert (out.status, out.detail) == ("skipped", "budget")
    assert judge.asked == 0 and response.results == results


class _Laya:
    backend = "laya"
    threshold = 0.5
    calibration_id = "c"
    top_k = 3


def _verdict() -> SufficiencyVerdict:
    return SufficiencyVerdict((0.9,), 0.5, "c")


@pytest.mark.parametrize("change", ["profile", "ids", "kinds", "judge_model"])
def test_a_verdict_is_reused_only_for_the_exact_binding(change) -> None:
    judge = _Laya()
    docs = ["a", "b"]
    first = memo.Binding("p", ("f1", "f2"), ("decision", "rule"), 5)
    memo.store(judge, "q", docs, _verdict(), binding=first)
    assert memo.lookup(judge, "q", docs, binding=first) is not None
    other = {
        "profile": memo.Binding("p2", first.fact_ids, first.kinds, 5),
        "ids": memo.Binding("p", ("f1", "f9"), first.kinds, 5),
        "kinds": memo.Binding("p", first.fact_ids, ("decision", "status"), 5),
        "judge_model": first,
    }[change]
    if change == "judge_model":
        judge.model = "another-model"
    assert memo.lookup(judge, "q", docs, binding=other) is None
    memo.clear(judge)


def test_a_verdict_about_a_memory_changed_since_is_not_reused() -> None:
    judge = _Laya()
    binding = memo.Binding("p", ("f1",), ("decision",), 7)
    memo.store(judge, "q", ["a"], _verdict(), binding=binding)
    seen: list = []

    def changed(since, ids):
        seen.append((since, ids))
        return True  # e.g. erased, withheld, re-kinded or consent-scoped since

    assert memo.lookup(judge, "q", ["a"], binding=binding, changed=changed) is None
    assert seen == [(7, ("f1",))]
    assert memo.lookup(judge, "q", ["a"], binding=binding) is None  # dropped for good
    memo.clear(judge)


def test_a_reused_verdict_is_told_apart_from_a_fresh_one(tmp_path) -> None:
    from tests.test_retrieval.cross_scope_fixture import REQ, build_store

    store = build_store(tmp_path / "m.db", n_global=0, n_shared=0, n_denied=0)
    judge = _Laya()
    judge.assess = lambda q, d, deadline=None: acs.JudgeOutcome(
        SufficiencyVerdict((0.9,) * len(d), 0.5, "c"), acs.STATUS_JUDGED)
    facts = [SimpleNamespace(fact_id=f, content=f, memory_kind="", profile_id=REQ)
             for f in store.ids("L")[:3]]
    response = SimpleNamespace(results=[SimpleNamespace(fact=f) for f in facts])
    engine = SimpleNamespace(_sufficiency_judge=judge, _db=store.db)
    first = run_answer_check(engine, "q", response, profile_id=REQ)
    again = run_answer_check(engine, "q", response, profile_id=REQ)
    assert (first.status, first.detail) == ("judged", "")
    assert (again.status, again.detail) == ("judged", "reused")
    store.db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?",
                     (facts[0].fact_id,))
    third = run_answer_check(engine, "q", response, profile_id=REQ)
    assert (third.status, third.detail) == ("judged", "")  # asked afresh, not reused
    memo.clear(judge)
