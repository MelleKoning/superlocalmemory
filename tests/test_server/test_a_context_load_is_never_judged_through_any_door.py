# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A recall that loads context is never judged — through the real doors.

The skip is one marker (``core.answer_check_scope``), a context variable. It
does not cross ``run_in_executor`` or a process, so each door is exercised for
real: the daemon's GET /recall route (TestClient, executor thread), the queue
consumer that serves the per-prompt hook (the daemon's in-process adapter), and
``slm session-context --full``. A recall that IS a question, through the same
route, is still judged.
"""

from __future__ import annotations

import contextlib
from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict

_VERDICT = SufficiencyVerdict((0.9,), 0.5, "test:judge")
_FACTS = [
    ("f-alpha", "the migration runs before the daemon accepts connections"),
    ("f-beta", "the daemon refuses a profile switch it cannot confirm locally"),
    ("f-gamma", "a withheld row is never shown as a memory"),
]
QUERY = "the daemon refuses a profile switch"


class _Judge:
    backend = "laya"
    top_k = 3

    def __init__(self) -> None:
        self.calls: list[str] = []

    def assess(self, query, documents, *, deadline=None):
        self.calls.append(query)
        return acs.JudgeOutcome(_VERDICT, acs.STATUS_JUDGED)

    def judge(self, query, documents):
        return self.assess(query, documents).verdict

    def shutdown(self) -> None: ...


class _ReorderingJudge(_Judge):
    backend = "jev"
    rerank_k = 20
    rerank_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.reorders = 0

    def rerank_and_judge(self, query, documents, *, deadline=None):
        self.reorders += 1
        return None


@pytest.fixture
def seeded(engine_with_mock_deps):
    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                       ("default", "default"))
    for fid, content in _FACTS:
        engine._db.execute(
            "INSERT INTO memories (memory_id, profile_id, content, session_id, speaker,"
            " role, created_at, metadata_json, scope) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"mem-{fid}", "default", content, "s", "user", "user",
             "2026-01-01T00:00:00Z", "{}", "personal"))
        engine._db.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, lifecycle)"
            " VALUES (?, ?, ?, ?, ?)", (fid, f"mem-{fid}", "default", content, "active"))
    return engine


def _with_judge(engine, judge):
    engine._ensure_init()
    engine._retrieval_engine._sufficiency_judge = judge
    return judge


def _client(engine) -> TestClient:
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app)


class TestTheRecallRoute:
    def test_a_skipped_recall_sends_nothing_to_the_judge(self, seeded) -> None:
        judge = _with_judge(seeded, _Judge())
        response = _client(seeded).get("/recall", params={"q": QUERY, "answer_check": "skip"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["results"], "skipping the check must not cost a single result"
        assert judge.calls == []
        assert body["answer_check_status"] == acs.STATUS_SKIPPED

    def test_a_question_through_the_same_route_is_judged(self, seeded) -> None:
        judge = _with_judge(seeded, _Judge())
        body = _client(seeded).get("/recall", params={"q": QUERY}).json()
        assert judge.calls == [QUERY]
        assert body["answer_check_status"] == acs.STATUS_JUDGED
        assert body["answer_confidence"] == pytest.approx(0.9)

    def test_no_reorder_asks_the_check_but_not_the_reorder(self, seeded) -> None:
        judge = _with_judge(seeded, _ReorderingJudge())
        body = _client(seeded).get(
            "/recall", params={"q": QUERY, "answer_check": "no_reorder"}).json()
        assert judge.reorders == 0 and judge.calls == [QUERY]
        assert body["answer_check_status"] == acs.STATUS_JUDGED

    def test_an_unknown_value_is_refused(self, seeded) -> None:
        _with_judge(seeded, _Judge())
        response = _client(seeded).get("/recall", params={"q": QUERY, "answer_check": "maybe"})
        assert response.status_code == 400


def test_the_queue_consumer_never_judges_the_per_prompt_hook(seeded) -> None:
    from superlocalmemory.core.queue_consumer import QueueConsumer
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    judge = _with_judge(seeded, _Judge())
    consumer = QueueConsumer(queue=None, pool=EngineRecallAdapter(seeded))
    import json

    result = json.loads(consumer._execute_recall(QUERY, 3, "hook-session"))
    assert result.get("results"), "the hook still gets its memories"
    assert judge.calls == []
    assert result["answer_check_status"] == acs.STATUS_SKIPPED


def test_session_context_full_never_judges(seeded, monkeypatch, capsys) -> None:
    from superlocalmemory.cli import commands
    from superlocalmemory.core import engine as engine_mod

    judge = _with_judge(seeded, _Judge())

    class _Same:
        def __init__(self, *_a, **_k) -> None: ...

        def initialize(self) -> None: ...

        def __getattr__(self, name):
            return getattr(seeded, name)

    monkeypatch.setattr(engine_mod, "MemoryEngine", _Same)
    with contextlib.suppress(SystemExit):
        commands.cmd_session_context(Namespace(full=True, json=True, query=QUERY))
    assert judge.calls == []
