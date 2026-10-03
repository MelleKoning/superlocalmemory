# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Narrowing a recall by project, by the agent that saved it, or by what it is about."""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.retrieval.facets import Facets, list_facets, matching_fact_ids
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    FactType,
    MemoryRecord,
    RecallResponse,
    RetrievalResult,
)


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    mgr.store_entity(CanonicalEntity(entity_id="e_pallavi", profile_id="default",
                                     canonical_name="Pallavi"))
    return mgr


def _save(db, content, *, project=None, agent=None, entities=()) -> str:
    meta = {k: v for k, v in (("project", project), ("agent_id", agent)) if v is not None}
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content, metadata=meta))
    return db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                                    fact_type=FactType.SEMANTIC,
                                    canonical_entities=list(entities)))


def test_project_filter_ignores_case_and_spaces(db) -> None:
    slm = _save(db, "Ship 4.1.19 with memory kinds.", project="SuperLocalMemory")
    other = _save(db, "Bounded loops need a gate.", project="bounded-loops")
    none = _save(db, "Untagged memory.")
    keep = matching_fact_ids(db, [slm, other, none], "default", Facets.of(project=" superlocalmemory "))
    assert keep == {slm}


def test_facets_combine_as_and(db) -> None:
    a = _save(db, "A", project="slm", agent="claude-desktop")
    b = _save(db, "B", project="slm", agent="grok")
    assert matching_fact_ids(db, [a, b], "default", Facets.of(project="slm", agent="grok")) == {b}


def test_about_finds_memories_that_mention_the_name(db) -> None:
    about = _save(db, "Pallavi reviews the release notes.", entities=("e_pallavi",))
    other = _save(db, "Nothing about people.")
    assert matching_fact_ids(db, [about, other], "default", Facets.of(about="pallavi")) == {about}
    assert matching_fact_ids(db, [about, other], "default", Facets.of(about="Nobody Known")) == set()


def test_no_facets_keeps_everything_and_blank_values_are_no_facet(db) -> None:
    ids = [_save(db, "x"), _save(db, "y", project="p")]
    assert matching_fact_ids(db, ids, "default", Facets.of()) == set(ids)
    assert Facets.of(project="  ", agent="", about=None).empty


def test_list_facets_counts_memories_per_project_and_agent(db) -> None:
    _save(db, "1", project="SLM", agent="claude")
    _save(db, "2", project="slm", agent="claude")
    _save(db, "3", project="loops", agent="grok")
    _save(db, "4", project="")
    out = list_facets(db, "default")
    assert out["projects"] == [{"name": "slm", "memories": 2}, {"name": "loops", "memories": 1}]
    assert out["agents"] == [{"name": "claude", "memories": 2}, {"name": "grok", "memories": 1}]


def _response():
    fact = AtomicFact(fact_id="f1", content="x", fact_type=FactType.SEMANTIC)
    return RecallResponse(query="q", results=[RetrievalResult(fact=fact, score=0.5)])


@pytest.mark.parametrize("facets,expect_key", [(None, False), (Facets.of(), False),
                                               (Facets.of(project="slm"), True)])
def test_facets_reach_retrieval_only_when_set(engine_with_mock_deps, monkeypatch,
                                              facets, expect_key) -> None:
    seen = {}

    def fake_recall(*args, **kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr(engine_with_mock_deps._retrieval_engine, "recall", fake_recall)
    engine_with_mock_deps.recall("what did we ship", fast=True, facets=facets)
    assert ("facets" in seen) is expect_key
    if expect_key:
        assert seen["facets"].project == "slm"


def test_http_recall_passes_facets_to_the_engine(engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    seen = {}

    def fake_engine_recall(*args, **kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr(engine_with_mock_deps, "recall", fake_engine_recall)
    with _client(engine_with_mock_deps) as client:
        r1 = client.get("/recall", params={"q": "decisions", "project": "SLM",
                                           "saved_by": "claude-desktop", "about": "Pallavi"})
        plain = dict(seen); seen.clear()
        r2 = client.get("/recall", params={"q": "decisions"})
        facets_page = client.get("/api/v3/facets")
    assert r1.status_code == 200, r1.text and r2.status_code == 200
    assert plain["facets"].as_dict() == {"project": "SLM", "agent": "claude-desktop",
                                         "about": "Pallavi"}
    assert "facets" not in seen
    assert facets_page.status_code == 200 and "projects" in facets_page.json()


def test_mcp_proxy_sends_only_the_facets_that_were_set(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=8765)
    proxy.recall("q", project="slm", saved_by="", about="Pallavi")
    assert paths, "the proxy did not call the daemon"
    assert "project=slm" in paths[0] and "about=Pallavi" in paths[0]
    assert "saved_by" not in paths[0]


def test_cli_recall_adds_the_facets_to_the_request(monkeypatch) -> None:
    from superlocalmemory.cli import commands, daemon

    paths = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    args = Namespace(query="what did we decide", limit=5, json=True, project="SLM",
                     saved_by="claude-desktop", about="", fast=None, window="", as_of="",
                     known_as_of="", valid_at="", include_unknown=False, include_global=None,
                     include_shared=None, session_id="", profile_id="")
    try:
        commands.cmd_recall(args)
    except SystemExit:
        pass
    assert paths, "the CLI did not call the daemon"
    assert "&project=SLM" in paths[0] and "&saved_by=claude-desktop" in paths[0]
    assert "&about=" not in paths[0]
