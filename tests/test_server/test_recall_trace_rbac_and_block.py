# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""POST /api/v3/recall/trace: READ when team accounts are on, an answer_check
block for the Answer Check tab's try-it panel, and dashboard-tagged recalls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import answer_check_history as h
from superlocalmemory.retrieval.answer_check_status import AnswerCheckTrace
from superlocalmemory.server.routes import v3_api
from superlocalmemory.storage.models import RecallResponse

from .test_answer_check_history_routes import FakeRbac

TRACE = "/api/v3/recall/trace"


class _Engine:
    def __init__(self) -> None:
        self.origins: list[str] = []
        self._config = SimpleNamespace(retrieval=SimpleNamespace())

    def recall(self, query, **kwargs):
        self.origins.append(h._ORIGIN.get())
        resp = RecallResponse(query=query, results=[])
        resp.answer_check_status = "judged"
        resp.abstained = True
        resp.abstention_reason = "judged_insufficient"
        resp.answer_confidence = 0.18
        resp.answer_check_trace = AnswerCheckTrace(
            detail="", backend="laya", threshold=0.5, reordered=False,
            retrieval_ms=812.3, judge_ms=201.0, total_ms=1013.3)
        return resp


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setattr("superlocalmemory.server.routes.helpers.get_active_profile",
                        lambda: "default")
    monkeypatch.setattr(v3_api, "_record_learning_signals", lambda *a, **k: None)
    application = FastAPI()
    application.state.engine = _Engine()
    application.include_router(v3_api.router)
    return application


def test_trace_requires_read_when_rbac_active(app) -> None:
    app.state.rbac = FakeRbac()
    outsider = TestClient(app, headers={"X-SLM-User-Session": "outsider-tok"})
    res = outsider.post(TRACE, json={"query": "q"})
    assert res.status_code == 403
    assert app.state.engine.origins == [], "refused before any recall ran"
    viewer = TestClient(app, headers={"X-SLM-User-Session": "viewer-tok"})
    assert viewer.post(TRACE, json={"query": "q"}).status_code == 200
    app.state.rbac = FakeRbac(require_login=True)
    assert TestClient(app).post(TRACE, json={"query": "q"}).status_code == 401


def test_trace_open_in_personal_mode(app) -> None:
    assert TestClient(app).post(TRACE, json={"query": "q"}).status_code == 200


def test_trace_answer_check_block_shape(app) -> None:
    body = TestClient(app).post(TRACE, json={"query": "q"}).json()
    assert body["answer_check"] == {
        "status": "judged", "detail": "", "judge": "laya", "abstained": True,
        "abstention_reason": "judged_insufficient", "answer_confidence": 0.18,
        "threshold": 0.5, "reordered": False, "retrieval_ms": 812.3, "judge_ms": 201.0,
        "total_ms": 1013.3, "ceiling_ms": 3000.0}
    assert body["abstained"] is True  # the existing envelope is untouched


def test_trace_block_without_a_trace_has_null_timings(app, monkeypatch) -> None:
    engine = app.state.engine
    real = engine.recall

    def no_trace(query, **kw):
        resp = real(query, **kw)
        resp.answer_check_trace = None
        return resp
    monkeypatch.setattr(engine, "recall", no_trace)
    block = TestClient(app).post(TRACE, json={"query": "q"}).json()["answer_check"]
    assert block["total_ms"] is None and block["judge_ms"] is None and block["judge"] == ""


def test_trace_tags_origin_dashboard(app) -> None:
    TestClient(app).post(TRACE, json={"query": "q"})
    assert app.state.engine.origins == ["dashboard"]
    assert h._ORIGIN.get() == ""  # nothing leaks into this thread


def test_trace_says_why_a_recall_was_not_checked(app, monkeypatch) -> None:
    """The Try-it panel shows ``answer_check_note`` under its verdict."""
    engine = app.state.engine
    real = engine.recall

    def skipped(query, **kw):
        resp = real(query, **kw)
        resp.answer_check_status, resp.answer_check_detail = "skipped", "budget"
        resp.abstained, resp.abstention_reason, resp.answer_confidence = False, None, None
        resp.answer_check_trace = AnswerCheckTrace(
            detail="budget", backend="laya", threshold=None, reordered=False,
            retrieval_ms=2950.0, judge_ms=0.0, total_ms=2951.0)
        return resp
    monkeypatch.setattr(engine, "recall", skipped)
    body = TestClient(app).post(TRACE, json={"query": "q"}).json()
    assert body["answer_check"]["detail"] == "budget"
    assert body["answer_check_reason"] == "budget" and body["answer_check_ran"] is False
    assert body["answer_check_note"].startswith("Answer check did not run")
    assert "only the verdict is missing" in body["answer_check_note"]
