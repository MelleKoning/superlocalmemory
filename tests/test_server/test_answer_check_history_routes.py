# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Answer Check history routes: scoped, guarded, and free of any text."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.access.rbac import Permission, Role, permissions_for_role
from superlocalmemory.core import answer_check_history as h
from superlocalmemory.core import answer_check_history_store as store
from superlocalmemory.server.routes import answer_check_history as routes
from superlocalmemory.storage import migration_runner as mr

from ..test_core.test_answer_check_history import make_response


class FakeRbac:
    """Users by session token; a role per (user, profile)."""

    def __init__(self, *, require_login: bool = False) -> None:
        self.users = {"viewer-tok": "viewer", "member-tok": "member", "admin-tok": "admin",
                      "outsider-tok": "outsider"}
        self.roles = {("viewer", "default"): Role.VIEWER, ("member", "default"): Role.MEMBER,
                      ("admin", "default"): Role.ADMIN}
        self._require_login = require_login

    def user_count(self) -> int:
        return len(self.users)

    def require_login(self) -> bool:
        return self._require_login

    def resolve_session(self, token):
        uid = self.users.get(token)
        return {"user_id": uid, "username": uid} if uid else None

    def has_permission(self, user_id, profile, permission: Permission) -> bool:
        role = self.roles.get((user_id, profile))
        return role is not None and permission in permissions_for_role(role)

    def get_role(self, user_id, profile):
        return self.roles.get((user_id, profile))


@pytest.fixture()
def learning_db(tmp_path):
    from superlocalmemory.storage import schema

    learning, memory = tmp_path / "learning.db", tmp_path / "memory.db"
    with sqlite3.connect(memory) as conn:
        schema.create_all_tables(conn)
    mr.apply_all(learning, memory)
    return learning


@pytest.fixture()
def app(learning_db, monkeypatch):
    store._reset_for_testing()
    h._reset_for_testing()
    routes._reset_for_testing()
    store._state["learning_db"] = learning_db
    h.enable(True)
    profile = {"name": "default"}
    monkeypatch.setattr(routes, "_profile", lambda: profile["name"])
    application = FastAPI()
    application.state.engine = SimpleNamespace(
        _config=SimpleNamespace(retrieval=SimpleNamespace(answer_check_history=True)))
    application.state.profile = profile
    application.include_router(routes.router)
    yield application
    store._reset_for_testing()
    h._reset_for_testing()
    routes._reset_for_testing()


def _record(n=1, profile="default", **kw):
    for _ in range(n):
        h.record_recall_verdict(make_response(**kw), profile_id=profile)


def _client(app, token: str | None = None, **kw) -> TestClient:
    client = TestClient(app, **kw)
    if token:
        client.headers["X-SLM-User-Session"] = token
    return client


READS = ["/api/v3/answer-check/history/live", "/api/v3/answer-check/history/summary",
         "/api/v3/answer-check/history"]


@pytest.mark.parametrize("path", READS)
def test_reads_require_read(app, path) -> None:
    app.state.rbac = FakeRbac()
    assert _client(app, "viewer-tok").get(path).status_code == 200
    assert _client(app, "outsider-tok").get(path).status_code == 403
    assert _client(app).get(path).status_code == 200            # owner, personal mode
    app.state.rbac = FakeRbac(require_login=True)
    routes._reset_for_testing()
    assert _client(app).get(path).status_code == 401            # owner must log in


def test_clear_requires_delete_and_credential(app) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    app.state.rbac = FakeRbac()
    token = ensure_install_token()
    _record(3)
    member = _client(app, "member-tok")
    member.headers["X-Install-Token"] = token
    assert member.delete("/api/v3/answer-check/history").status_code == 403
    remote = _client(app, "admin-tok", client=("10.1.2.3", 5000))
    remote.headers["X-Install-Token"] = token
    assert remote.delete("/api/v3/answer-check/history").status_code == 403
    no_cred = _client(app, "admin-tok")
    assert no_cred.delete("/api/v3/answer-check/history").status_code == 403
    admin = _client(app, "admin-tok")
    admin.headers["X-Install-Token"] = token
    res = admin.delete("/api/v3/answer-check/history")
    assert res.status_code == 200, res.text
    assert h.recent("default", after_seq=0, limit=50)[0] == []


def test_routes_never_return_other_profile(app) -> None:
    _record(3, "default")
    _record(4, "other")
    store.flush_once()
    _record(2, "other")                                   # unsaved, in the ring
    client = _client(app)
    live = client.get("/api/v3/answer-check/history/live").json()
    page = client.get("/api/v3/answer-check/history").json()
    summary = client.get("/api/v3/answer-check/history/summary").json()
    assert len(live["items"]) == 3 and len(page["items"]) == 3
    assert summary["counts"]["total"] == 3


def test_feed_payload_has_no_query_or_content(app) -> None:
    resp = make_response()
    resp.query = "SECRET-Q"
    resp.results = [SimpleNamespace(fact=SimpleNamespace(content="SECRET-M", fact_id="FACT-9"))]
    h.record_recall_verdict(resp, profile_id="default")
    client = _client(app)
    bodies = [client.get(p).text for p in READS]
    store.flush_once()
    routes._reset_for_testing()
    bodies += [client.get(p).text for p in READS]
    for body in bodies:
        assert "SECRET-Q" not in body and "SECRET-M" not in body and "FACT-9" not in body
    item = json.loads(bodies[0])["items"][0]
    assert set(item) == {"seq", "id", "at", "outcome", "status", "detail", "judge", "origin",
                         "abstained", "abstention_reason", "answer_confidence", "threshold",
                         "reordered", "result_count", "query_type", "retrieval_ms",
                         "judge_ms", "total_ms", "over_ceiling", "embed_ms", "rerank_ms"}


def test_cursor_validation_400(app) -> None:
    client = _client(app)
    for bad in ("x", "12:ABC", "1:" + "g" * 32, "9" * 16 + ":" + "a" * 32):
        res = client.get("/api/v3/answer-check/history", params={"cursor": bad})
        assert res.status_code == 400 and res.json() == {"error": "That page link is not valid."}
        routes._reset_for_testing()


def test_pagination_walks_saved_and_unsaved(app) -> None:
    _record(7)
    store.flush_once()
    _record(3)                                            # unsaved, newest
    client = _client(app)
    seen, cursor = [], None
    while True:
        params = {"limit": 4, **({"cursor": cursor} if cursor else {})}
        body = client.get("/api/v3/answer-check/history", params=params).json()
        routes._reset_for_testing()
        seen += [item["id"] for item in body["items"]]
        cursor = body["next_cursor"]
        if not cursor:
            break
    assert len(seen) == 10 and len(set(seen)) == 10


def test_rate_limit_429_with_retry_after(app) -> None:
    client = _client(app)
    codes = [client.get("/api/v3/answer-check/history/summary").status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and 429 in codes
    res = client.get("/api/v3/answer-check/history/summary")
    assert res.status_code == 429 and int(res.headers["Retry-After"]) >= 1
    assert res.json()["retry_after_ms"] > 0


def test_history_disabled_returns_enabled_false(app) -> None:
    app.state.engine._config.retrieval.answer_check_history = False
    client = _client(app)
    for path in READS:
        res = client.get(path)
        assert res.status_code == 200 and res.json()["enabled"] is False


def test_summary_excludes_dashboard_by_default(app) -> None:
    _record(5)
    with h.origin("dashboard"):
        _record(7, abstained=True, status="judged")
    client = _client(app)
    body = client.get("/api/v3/answer-check/history/summary").json()
    assert body["counts"]["total"] == 5 and body["include_dashboard"] is False
    both = client.get("/api/v3/answer-check/history/summary",
                      params={"include_dashboard": "true"}).json()
    assert both["counts"]["total"] == 12
    live = client.get("/api/v3/answer-check/history/live").json()
    assert sum(1 for i in live["items"] if i["origin"] == "dashboard") == 7


def test_saved_view_runs_are_labelled_by_surface_and_counted(app) -> None:
    """4.1.21 (#113): a view run says where it came from. Unlike a dashboard
    test it is a real question, so it counts in the summary."""
    _record(2)
    for origin, n in (("view-dashboard", 1), ("view-cli", 2), ("view-mcp", 3)):
        with h.origin(origin):
            _record(n)
    store.flush_once()
    with h.origin("view-mcp"):
        _record(1)                                        # unsaved, in the ring
    client = _client(app)
    live = client.get("/api/v3/answer-check/history/live").json()["items"]
    page = client.get("/api/v3/answer-check/history").json()["items"]
    for items in (live, page):
        labels = sorted(i["origin"] for i in items)
        assert labels.count("view-dashboard") == 1 and labels.count("view-cli") == 2
        assert labels.count("other") == 2
    assert sorted(i["origin"] for i in live).count("view-mcp") == 4
    summary = client.get("/api/v3/answer-check/history/summary").json()
    assert summary["counts"]["total"] == 9


def test_summary_shape_and_recording(app) -> None:
    _record(25)
    _record(5, status="judged", abstained=True)
    body = _client(app).get("/api/v3/answer-check/history/summary",
                            params={"window": "24h"}).json()
    assert body["abstention"]["n"] == 30 and body["abstention"]["enough"] is True
    assert body["latency"]["ceiling_ms"] == 3000.0
    rec = body["recording"]
    assert rec["retention_days"] == 30 and rec["max_rows"] == 10_000
    assert rec["since_start"]["recorded"] == 30 and rec["unsaved"] == 30


def test_profile_switch_feed_follows_active_profile(app) -> None:
    _record(2, "default")
    _record(3, "work")
    client = _client(app)
    assert len(client.get("/api/v3/answer-check/history/live").json()["items"]) == 2
    app.state.profile["name"] = "work"
    assert len(client.get("/api/v3/answer-check/history/live").json()["items"]) == 3


def _staged(embed: float | None, rerank: float | None):
    resp = make_response()
    resp.stage_ms = {k: v for k, v in (("query_embedding", embed), ("rerank", rerank),
                                       ("channels", 650.0)) if v is not None}
    return resp


def test_stage_timings_survive_the_save_and_reach_the_feed_and_summary(app) -> None:
    """Where a recall's time went is saved with the check and read back from the
    database -- not only from the in-memory ring -- in the feed and summary."""
    for embed, rerank in ((40.0, 30.0), (60.0, 50.0), (900.5, 70.0)):
        h.record_recall_verdict(_staged(embed, rerank), profile_id="default")
    h.record_recall_verdict(_staged(None, None), profile_id="default")
    store.flush_once()
    h._reset_for_testing()                               # the ring is gone: database only
    h.enable(True)
    client = _client(app)
    items = client.get("/api/v3/answer-check/history").json()["items"]
    assert sorted((i["embed_ms"], i["rerank_ms"]) for i in items if i["embed_ms"]) == [
        (40.0, 30.0), (60.0, 50.0), (900.5, 70.0)]
    assert [i for i in items if i["embed_ms"] is None][0]["rerank_ms"] is None
    stages = client.get("/api/v3/answer-check/history/summary").json()["latency"]["stages"]
    assert stages["embed"] == {"n": 3, "p50": 60.0, "p95": 900.5}
    assert stages["rerank"] == {"n": 3, "p50": 50.0, "p95": 70.0}
    assert stages["retrieval"]["n"] == 4 and stages["retrieval"]["p50"] == 700.0
    with sqlite3.connect(store._state["learning_db"]) as conn:
        cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(answer_check_events)")}
    assert cols["embed_ms"] == cols["rerank_ms"] == "REAL"        # numbers only


def test_a_stage_timing_that_is_not_a_number_is_not_recorded() -> None:
    resp = make_response()
    resp.stage_ms = {"query_embedding": "SECRET-Q", "rerank": float("nan")}
    ev = h.event_from_response(resp, "default", now_ms=1, origin_name="")
    assert ev is not None and ev.embed_ms is None and ev.rerank_ms is None
    resp.stage_ms = {"query_embedding": 10**9, "rerank": -5}
    ev = h.event_from_response(resp, "default", now_ms=1, origin_name="")
    assert ev.embed_ms == 600_000.0 and ev.rerank_ms == 0.0
