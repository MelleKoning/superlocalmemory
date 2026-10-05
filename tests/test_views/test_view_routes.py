# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saved views over HTTP: who may do what, profile scope, refusals, and that a
run is recall's own answer, the same twice (issue #113)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import views as routes
from superlocalmemory.views import ViewStore

from ..test_server.test_answer_check_history_routes import FakeRbac
from ._store import learning_db


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    learning_db(tmp_path)
    return tmp_path


@pytest.fixture()
def profile(monkeypatch):
    current = {"name": "default"}
    monkeypatch.setattr(routes, "_profile", lambda: current["name"])
    return current


@pytest.fixture()
def app(data_dir, profile):
    application = FastAPI()
    application.state.engine = None
    application.include_router(routes.router)
    return application


def _client(app, session: str | None = None, *, write: bool = False, **kw) -> TestClient:
    client = TestClient(app, **kw)
    if session:
        client.headers["X-SLM-User-Session"] = session
    if write:
        from superlocalmemory.core.security_primitives import ensure_install_token

        client.headers["X-Install-Token"] = ensure_install_token()
    return client


def _create(client, **body):
    return client.post("/api/v3/views", json={"name": "Work", "query": "what shipped",
                                              **body})


class TestCrud:
    def test_create_list_show_rename_delete(self, app) -> None:
        client = _client(app, write=True)
        made = _create(client, filters={"kind": "decision"}, limit=5)
        assert made.status_code == 200, made.text
        assert made.json()["view"]["filters"] == {"kind": "decision"}
        listing = client.get("/api/v3/views").json()
        assert [v["name"] for v in listing["views"]] == ["Work"]
        assert listing["limits"]["max_results"] == 50
        shown = client.get("/api/v3/views/show", params={"name": "work"}).json()
        assert shown["view"]["query"] == "what shipped"
        renamed = client.post("/api/v3/views/rename", json={"name": "Work", "new_name": "a/b"})
        assert renamed.status_code == 200 and renamed.json()["view"]["name"] == "a/b"
        assert client.get("/api/v3/views/show", params={"name": "a/b"}).status_code == 200
        gone = client.post("/api/v3/views/delete", json={"name": "a/b"})
        assert gone.status_code == 200 and "No memory was changed" in gone.json()["message"]
        assert client.get("/api/v3/views").json()["views"] == []

    @pytest.mark.parametrize("body, code", [
        ({"name": ""}, "invalid_view_name"),
        ({"query": "x" * 1001}, "invalid_view_query"),
        ({"limit": 0}, "invalid_view_limit"),
        ({"filters": {"project": "slm"}}, "unknown_view_filter"),
        ({"filters": {"window": "fortnight"}}, "invalid_view_filter"),
    ])
    def test_bad_input_is_a_422_with_a_code(self, app, body, code) -> None:
        res = _create(_client(app, write=True), **body)
        assert res.status_code == 422
        assert res.json()["detail"]["code"] == code

    def test_unknown_fields_are_refused(self, app) -> None:
        res = _create(_client(app, write=True), prompt="summarise everything")
        assert res.status_code == 422

    def test_duplicate_and_missing(self, app) -> None:
        client = _client(app, write=True)
        _create(client)
        dup = _create(client)
        assert dup.status_code == 409 and dup.json()["code"] == "view_exists"
        missing = client.get("/api/v3/views/run", params={"name": "nope"})
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "view_not_found"

    def test_a_store_that_is_not_ready_says_so(self, app, data_dir) -> None:
        import sqlite3

        with sqlite3.connect(data_dir / "learning.db") as conn:
            conn.execute("DROP TABLE saved_views")
        res = _client(app).get("/api/v3/views")
        assert res.status_code == 409 and res.json()["code"] == "views_unavailable"


class TestAuthorization:
    def test_a_write_needs_a_product_credential(self, app) -> None:
        assert _create(_client(app)).status_code == 403
        assert _client(app).post("/api/v3/views/delete", json={"name": "x"}).status_code == 403

    def test_a_viewer_can_read_and_run_but_not_change(self, app, data_dir) -> None:
        app.state.rbac = FakeRbac()
        ViewStore(data_dir / "learning.db").create("default", name="Work", query="q")
        viewer = _client(app, "viewer-tok", write=True)
        assert viewer.get("/api/v3/views").status_code == 200
        assert viewer.get("/api/v3/views/show", params={"name": "Work"}).status_code == 200
        assert _create(viewer, name="Mine").status_code == 403
        assert viewer.post("/api/v3/views/rename",
                           json={"name": "Work", "new_name": "x"}).status_code == 403
        assert viewer.post("/api/v3/views/delete", json={"name": "Work"}).status_code == 403
        member = _client(app, "member-tok", write=True)
        assert _create(member, name="Mine").status_code == 200

    def test_a_stranger_to_the_workspace_cannot_read(self, app) -> None:
        app.state.rbac = FakeRbac()
        outsider = _client(app, "outsider-tok")
        for path in ("/api/v3/views", "/api/v3/views/run?name=Work",
                     "/api/v3/views/show?name=Work"):
            assert outsider.get(path).status_code == 403, path

    def test_company_mode_requires_a_login(self, app) -> None:
        app.state.rbac = FakeRbac(require_login=True)
        assert _client(app).get("/api/v3/views").status_code == 401

    def test_views_are_classed_as_sensitive_reads(self) -> None:
        from superlocalmemory.server.read_gates import is_sensitive_dashboard_read

        for path in ("/api/v3/views", "/api/v3/views/run", "/api/v3/views/show",
                     "/api/summary", "/api/summary/projects", "/api/summary/sessions"):
            assert is_sensitive_dashboard_read("GET", path), path


class TestProfileScope:
    def test_each_profile_sees_only_its_own_views(self, app, profile) -> None:
        client = _client(app, write=True)
        _create(client, name="Alice plan", query="acquisition")
        profile["name"] = "bob"
        assert client.get("/api/v3/views").json()["views"] == []
        assert client.get("/api/v3/views/run",
                          params={"name": "Alice plan"}).status_code == 404
        assert client.post("/api/v3/views/delete",
                           json={"name": "Alice plan"}).status_code == 404
        profile["name"] = "default"
        assert [v["name"] for v in client.get("/api/v3/views").json()["views"]] \
            == ["Alice plan"]


class TestRunIsRecall:
    def test_the_view_arguments_reach_engine_recall(self, app, data_dir) -> None:
        calls: list[tuple] = []

        def recall(query, **kw):
            calls.append((query, kw))
            return SimpleNamespace(results=[], query_type="semantic",
                                   retrieval_time_ms=1.0, no_confident_match=True)
        app.state.engine = SimpleNamespace(recall=recall, _config=SimpleNamespace(),
                                           profile_id="default")
        ViewStore(data_dir / "learning.db").create(
            "default", name="Work", query="what shipped", limit=7,
            filters={"kind": "decision", "window": "7d"})
        client = _client(app)
        first = client.get("/api/v3/views/run", params={"name": "Work"})
        client.get("/api/v3/views/run", params={"name": "Work"})
        assert first.status_code == 200, first.text
        (q1, kw1), (_q2, kw2) = calls
        assert q1 == "what shipped" and kw1["limit"] == 7 and kw1["window"] == "7d"
        assert kw1["facets"].kind == "decision" and kw1["fast"] is True
        assert kw1["profile_id"] == "default"
        # A view is not a conversation: continuity must ignore its runs.
        from superlocalmemory.core.session_identity import is_conversation

        assert kw1["session_id"].startswith("view:")
        assert is_conversation(kw1["session_id"], "default") is False
        assert first.json()["no_confident_match"] is True

    def test_same_view_twice_gives_the_same_ids_in_the_same_order(
            self, app, data_dir, engine_with_mock_deps) -> None:
        from tests.conftest import force_sync_enrichment

        engine = force_sync_enrichment(engine_with_mock_deps)
        for text in ("The widget release shipped on Tuesday after both audits.",
                     "We decided the widget cache lives in learning.db.",
                     "Widget docs were updated for the release.",
                     "The cat sat on the mat."):
            engine.store(text)
        app.state.engine = engine
        ViewStore(data_dir / "learning.db").create(
            "default", name="Widget", query="widget release", limit=5)
        client = _client(app)
        from superlocalmemory.core.working_memory import registry_size

        before = registry_size()
        runs = [client.get("/api/v3/views/run", params={"name": "Widget"}).json()
                for _ in range(3)]
        ids = [run["result_ids"] for run in runs]
        assert ids[0], runs[0]
        assert ids[0] == ids[1] == ids[2]
        assert all(r["fact_id"] and r["rank"] == i
                   for i, r in enumerate(runs[0]["results"], start=1))
        # Runs hold no conversation working set: they cannot bias each other
        # or push a real conversation out of the registry.
        from superlocalmemory.core.working_memory import registry_size

        assert registry_size() == before
        # Every id is a real memory the owner can open.
        stored = {f.fact_id for f in engine._db.get_all_facts("default")}
        assert set(ids[0]) <= stored
