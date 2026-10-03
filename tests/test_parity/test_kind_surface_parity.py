# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One behaviour, three doors (LLD/WP8 4.1.19): the memory-kind surface
(set/status/review/confirm, and the ``kind`` filter on recall/list) answers
the same way whether it is reached through an MCP tool function, the CLI
(with ``daemon_request`` bound to a real FastAPI ``TestClient``), or raw
HTTP. "Identical" means the same ``kind_fields`` and the same set of fact
ids — not byte-identical envelopes, since each surface wraps its answer in
its own transport shape (MCP's ``{"success": ...}``, the CLI's
``json_print`` envelope, HTTP's own response body).
"""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace

from fastapi.testclient import TestClient

from superlocalmemory.cli import kinds_cmd
from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound
from superlocalmemory.storage.models import AtomicFact, FactType
from tests.test_server.test_memory_kinds_routes import _client, _fact, _session

KIND_KEYS = ("memory_kind", "memory_kind_label", "memory_kind_state",
            "memory_kind_source", "memory_kind_confidence")


def _kind_subset(d: dict) -> dict:
    return {k: d.get(k) for k in KIND_KEYS}


# -- shared plumbing ------------------------------------------------------------


class _Server:
    def __init__(self) -> None:
        self.captured: dict = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.captured[fn.__name__] = fn
            return fn
        return deco


def _kind_tools():
    from superlocalmemory.mcp import tools_kinds

    server = _Server()
    tools_kinds.register_kind_tools(server, lambda: None)
    return server.captured


def _bind_daemon_request(monkeypatch, tc) -> None:
    """Bind BOTH ``kinds_cmd.daemon_request`` (CLI) and ``cli.daemon.daemon_request``
    (MCP's inline import) to the same TestClient, so all three surfaces read
    from the identical underlying app+db."""

    def fake_daemon_request(method, path, body=None, *, preserve_conflict=False,
                            preserve_not_found=False, preserve_unprocessable=False, **_kw):
        import os

        # Matches cli.daemon.daemon_request: an opted-in caller session rides
        # along as a header, never silently dropped (needed so an MCP call
        # made under SLM_USER_SESSION is authenticated the same way the real
        # client would authenticate it).
        session = os.environ.get("SLM_USER_SESSION", "").strip()
        headers = {"X-SLM-User-Session": session} if session else {}
        response = tc.request(method, path, json=body, headers=headers)
        if response.status_code in (401, 403):
            # Matches cli.daemon.daemon_request: a refusal is an answer, not an
            # outage, on every door (L3-04) — never collapsed to None here.
            from superlocalmemory.cli.daemon import DaemonRefused

            raise DaemonRefused(response.status_code, path)
        if response.status_code == 409 and preserve_conflict:
            raise DaemonConflict(response.json().get("detail", ""))
        if response.status_code == 404 and preserve_not_found:
            raise DaemonNotFound(404, "not_found", "daemon returned 404", path)
        if response.status_code >= 400:
            return None
        return response.json()

    monkeypatch.setattr(kinds_cmd, "daemon_request", fake_daemon_request)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", fake_daemon_request)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)


def _cli_json(capsys, fn, *a, **k) -> dict:
    fn(*a, **k)
    return json.loads(capsys.readouterr().out)


def _kinds_args(**kw) -> Namespace:
    base = {"json": True, "kinds_command": None, "backfill_command": None}
    base.update(kw)
    return Namespace(**base)


# -- test_set_kind_identical ----------------------------------------------------


def test_set_kind_identical(tmp_path, monkeypatch, capsys) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    f_mcp = _fact(db, "An MCP-set decision.")
    f_cli = _fact(db, "A CLI-set decision.")
    f_http = _fact(db, "An HTTP-set decision.")
    _bind_daemon_request(monkeypatch, tc)

    mcp_out = asyncio.run(_kind_tools()["set_memory_kind"](f_mcp, "decision"))
    assert mcp_out["success"] is True

    cli_out = _cli_json(capsys, kinds_cmd.cmd_kinds,
                        _kinds_args(kinds_command="set", fact_id=f_cli, kind="decision"))
    assert cli_out["success"] is True

    http_out = tc.patch(f"/api/memory-kinds/fact/{f_http}", json={"kind": "decision"})
    assert http_out.status_code == 200, http_out.text

    assert _kind_subset(mcp_out) == _kind_subset(cli_out["data"]) == _kind_subset(http_out.json())
    assert mcp_out["memory_kind"] == "decision"


# -- test_daemon_refusal_identical (L3-04) ---------------------------------


def test_daemon_refusal_identical(tmp_path, monkeypatch) -> None:
    """A role without WRITE is refused identically on HTTP and MCP.

    Before L3-04, MCP mapped this 403 to DAEMON_UNAVAILABLE/retryable=True —
    an outage a caller might retry forever — while HTTP correctly answered
    403. Both doors now agree: not authorized, not retryable.
    """
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    tc, app, db = _client(tmp_path, monkeypatch)
    fid = _fact(db, "Never push to main.")
    viewer = _session(app, "vera", "viewer")
    app.state.rbac.set_require_login(True)
    monkeypatch.setenv("SLM_USER_SESSION", viewer["X-SLM-User-Session"])
    _bind_daemon_request(monkeypatch, tc)

    http_out = tc.patch(f"/api/memory-kinds/fact/{fid}", json={"kind": "rule"},
                        headers=viewer)
    assert http_out.status_code == 403, http_out.text

    mcp_out = asyncio.run(_kind_tools()["set_memory_kind"](fid, "rule"))
    assert mcp_out["success"] is False
    assert mcp_out["code"] == "NOT_AUTHORIZED"
    assert mcp_out["retryable"] is False


# -- test_status_identical -------------------------------------------------------


def test_status_identical(tmp_path, monkeypatch, capsys) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "Never push to main.", kind="rule", source="user")
    _bind_daemon_request(monkeypatch, tc)

    mcp_out = asyncio.run(_kind_tools()["memory_kinds_status"]())
    cli_out = _cli_json(capsys, kinds_cmd.cmd_kinds, _kinds_args(kinds_command="status"))
    http_out = tc.get("/api/memory-kinds/status").json()

    for payload in (mcp_out, cli_out["data"], http_out):
        assert payload["schema_ready"] is True
        assert payload["enabled"] == http_out["enabled"]
        assert payload["counts"] == http_out["counts"]
        assert payload["backend"] == http_out["backend"]


# -- test_bad_kind_refused_identically --------------------------------------


def test_bad_kind_refused_identically(tmp_path, monkeypatch, capsys) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    fid = _fact(db, "Something.")
    _bind_daemon_request(monkeypatch, tc)

    mcp_out = asyncio.run(_kind_tools()["set_memory_kind"](fid, "not-a-real-kind"))
    assert mcp_out["success"] is False and mcp_out["code"] == "INVALID_KIND"

    import pytest
    with pytest.raises(SystemExit) as exited:
        kinds_cmd.cmd_kinds(
            _kinds_args(kinds_command="set", fact_id=fid, kind="not-a-real-kind"),
        )
    assert exited.value.code == 2

    http_out = tc.patch(f"/api/memory-kinds/fact/{fid}", json={"kind": "not-a-real-kind"})
    assert http_out.status_code == 422

    # Nothing was mutated by any of the three refused attempts.
    row = dict(db.execute(
        "SELECT memory_kind FROM atomic_facts WHERE fact_id = ?", (fid,))[0])
    assert row["memory_kind"] is None


# -- test_recall_kind_filter_identical ------------------------------------------


def _recall_result(fact_id, content, *, kind, source):
    fact = AtomicFact(fact_id=fact_id, memory_id=f"m-{fact_id}", content=content,
                      fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source=source)
    return SimpleNamespace(fact=fact, score=0.9, relevance_score=0.9, confidence=0.9,
                           memory_confidence=0.9, ranking_score=0.9, rank_position=1,
                           trust_score=0.5, channel_scores={}, evidence_chain=[])


_RECALL_CANDIDATES = [
    _recall_result("f1", "a decision", kind="decision", source="user"),
    _recall_result("f2", "a fact", kind="semantic", source="user"),
    _recall_result("f3", "another decision", kind="decision", source="user"),
]


def _facet_filtered_recall(*_args, **kwargs):
    """Stand-in for MemoryEngine.recall(): applies the ``kind`` facet the same
    way the real RetrievalEngine does (matching_fact_ids, BEFORE the answer
    check — see test_the_kind_facet_runs_before_the_judge.py), so this proves
    the three surfaces agree on what reaches the engine, not just on the
    shape of a canned response.
    """
    facets = kwargs.get("facets")
    if facets is not None and not facets.empty and facets.kind:
        kept = [r for r in _RECALL_CANDIDATES if r.fact.memory_kind == facets.kind]
    else:
        kept = _RECALL_CANDIDATES
    return SimpleNamespace(results=kept, query="q", query_type="lookup",
                           retrieval_time_ms=1.0, channel_weights={},
                           total_candidates=len(kept), no_confident_match=False)


def test_recall_kind_filter_identical(engine_with_mock_deps, monkeypatch, capsys) -> None:
    from superlocalmemory.cli import commands, daemon
    from superlocalmemory.mcp import _daemon_proxy, tools_core
    from tests.test_server.test_canonical_remember_route import _client as recall_client

    monkeypatch.setattr(engine_with_mock_deps, "recall", _facet_filtered_recall)

    with recall_client(engine_with_mock_deps) as client:
        http_body = client.get("/recall", params={"q": "decisions", "kind": "decision"}).json()

        # MCP: route the daemon proxy at the SAME TestClient.
        monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)

        def _mcp_daemon_request(method, path, body=None, **_kw):
            resp = client.request(method, path, json=body)
            return resp.json() if resp.status_code < 400 else None

        monkeypatch.setattr(daemon, "daemon_request", _mcp_daemon_request)
        # Sidestep choose_pool()'s own daemon-discovery (port files, health
        # probes) — it is not what this test is about — and bind it straight
        # to a proxy that talks to the SAME TestClient the other two surfaces
        # use, through the daemon_request patched above.
        monkeypatch.setattr(_daemon_proxy, "choose_pool",
                            lambda: _daemon_proxy.DaemonPoolProxy(port=1))
        server = _Server()
        tools_core.register_core_tools(server, lambda: engine_with_mock_deps)
        mcp_body = asyncio.run(server.captured["recall"]("decisions", kind="decision"))

        # CLI: same TestClient, through cmd_recall's query-string construction.
        # (daemon.is_daemon_running / daemon_request are already bound above.)
        cli_out = _cli_json(
            capsys, commands.cmd_recall,
            Namespace(query="decisions", limit=10, json=True, project="", saved_by="",
                     about="", fast=None, window="", as_of="", known_as_of="", valid_at="",
                     include_unknown=False, include_global=None, include_shared=None,
                     session_id="", profile_id="", kind="decision"),
        )

    def _ids(results):
        return {r["fact_id"] for r in results}

    def _kinds(results):
        return {r["fact_id"]: _kind_subset(r) for r in results}

    assert _ids(http_body["results"]) == {"f1", "f3"}
    assert _ids(mcp_body["results"]) == {"f1", "f3"}
    assert _ids(cli_out["data"]["results"]) == {"f1", "f3"}
    assert _kinds(http_body["results"]) == _kinds(mcp_body["results"]) == _kinds(
        cli_out["data"]["results"])


# -- test_list_kind_filter_identical ---------------------------------------


class _FakeEngine:
    def __init__(self, db) -> None:
        self._db = db
        self.profile_id = "default"

    def initialize(self) -> None:
        pass

    def recall(self, *args, **kwargs):
        """None (not raising) is exactly how /api/search reads a recall that
        timed out or came back empty-handed -- it falls through to the
        degraded-lexical DB fallback deterministically, with no timing race."""
        return None


def test_list_kind_filter_identical(tmp_path, monkeypatch, capsys) -> None:
    from superlocalmemory.cli import commands
    from superlocalmemory.core import engine as engine_mod
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import MemoryRecord

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)

    def _save(content, *, kind=None, source=None):
        memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
        fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                          fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source=source)
        return db.store_fact(fact)

    _save("rule one", kind="rule", source="user")
    _save("a decision", kind="decision", source="user")
    _save("rule two", kind="rule", source="user")

    fake_engine = _FakeEngine(db)
    monkeypatch.setattr(engine_mod, "MemoryEngine", lambda *a, **k: fake_engine)

    server = _Server()
    from superlocalmemory.mcp import tools_core
    tools_core.register_core_tools(server, lambda: fake_engine)
    mcp_out = asyncio.run(server.captured["list_recent"](limit=10, kind="rule"))

    cli_out = _cli_json(capsys, commands.cmd_list, Namespace(json=True, limit=10, kind="rule"))

    # HTTP: /list (L3-20, L3-06) reads through the same fake_engine._db.
    from superlocalmemory.server.unified_daemon import create_app

    app = create_app()
    app.state.engine = fake_engine
    http_out = TestClient(app).get("/list", params={"kind": "rule"}).json()

    def _by_content(results):
        return {r["content"]: _kind_subset(r) for r in results}

    assert set(_by_content(mcp_out["results"])) == {"rule one", "rule two"}
    assert _by_content(mcp_out["results"]) == _by_content(cli_out["data"]["results"]) == \
        _by_content(http_out["results"])


# -- test_search_kind_filter_identical --------------------------------------


def test_search_kind_filter_identical(tmp_path, monkeypatch, capsys) -> None:
    """``search`` (MCP) and POST /api/search (HTTP) share
    core.kind_query.search_facts / the same post-filter contract. The CLI has
    no dedicated ``search`` verb (``recall`` is its multi-channel search,
    covered by test_recall_kind_filter_identical above)."""
    from superlocalmemory.core import engine as engine_mod
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import MemoryRecord

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)

    def _save(content, *, kind=None, source=None):
        memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
        fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                          fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source=source)
        return db.store_fact(fact)

    _save("a rule about main", kind="rule", source="user")
    _save("a decision about main", kind="decision", source="user")

    fake_engine = _FakeEngine(db)
    monkeypatch.setattr(engine_mod, "MemoryEngine", lambda *a, **k: fake_engine)

    server = _Server()
    from superlocalmemory.mcp import tools_core
    tools_core.register_core_tools(server, lambda: fake_engine)
    mcp_out = asyncio.run(server.captured["search"]("main", limit=10, kind="rule"))

    from superlocalmemory.server.unified_daemon import create_app

    app = create_app()
    app.state.engine = fake_engine  # .recall() returns None -> degraded-lexical fallback
    http_out = TestClient(app).post(
        "/api/search", json={"query": "main", "limit": 10, "kind": "rule"}).json()

    def _by_content(results):
        return {r["content"]: _kind_subset(r) for r in results}

    assert set(_by_content(mcp_out["results"])) == {"a rule about main"}
    assert set(_by_content(http_out["results"])) == {"a rule about main"}
