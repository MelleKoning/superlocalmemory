# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``POST /session/open`` warms a session; it does not ask a question.

Its recall ("project context <project name>") must run with the answer
check skipped, so with the online check on nothing is judged, sent or
billed. It runs off the event loop, keeps its size bounded, and its failure
message never quotes an exception (which can carry a path or a memory).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.core.answer_check_scope import answer_check_skipped


class _Engine:
    profile_id = "default"
    _profile_id = "default"

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self._fail = fail

    def recall(self, query, limit=10, agent_id="", facets=None):
        import threading

        self.calls.append({"query": query, "limit": limit, "skipped": answer_check_skipped(),
                           "thread": threading.current_thread().name, "facets": facets})
        if self._fail:
            raise RuntimeError("boom at /Users/someone/secret-project with memory text")
        return SimpleNamespace(results=[1, 2, 3])


@pytest.fixture
def client_and_engine():
    from superlocalmemory.server.unified_daemon import create_app

    def build(fail: bool = False):
        app = create_app()
        engine = _Engine(fail=fail)
        app.state.engine = engine
        return TestClient(app), engine
    return build


def test_session_open_recall_skips_the_answer_check(client_and_engine):
    client, engine = client_and_engine()
    r = client.post("/session/open", json={"project_path": "/Users/someone/private-project"})
    assert r.status_code == 200, r.text
    assert r.json()["warmed"] == 3
    assert engine.calls and engine.calls[0]["skipped"] is True
    # 4.1.21 (#150): the project is preferred, and named - not its whole path.
    assert engine.calls[0]["query"] == "project context private-project"
    assert engine.calls[0]["facets"].prefer_project == "/Users/someone/private-project"


def test_session_open_bounds_its_size(client_and_engine):
    client, engine = client_and_engine()
    client.post("/session/open", json={"query": "x", "max_results": 10_000_000})
    client.post("/session/open", json={"query": "x", "max_results": -5})
    assert [c["limit"] for c in engine.calls] == [50, 1]


def test_a_failed_warm_up_never_quotes_the_exception(client_and_engine):
    client, _engine = client_and_engine(fail=True)
    r = client.post("/session/open", json={"query": "x"})
    assert r.status_code == 200
    body = r.json()
    assert body["warmed"] == 0
    assert "secret-project" not in r.text and "memory text" not in r.text
