# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L3-06: no HTTP 500 carries ``str(exc)`` — a local path, an internal
attribute name or any other exception text must never reach the caller, and
the server log carries the detail instead. Also covers /list, which
previously 500'd on every call because it called a ``MemoryEngine`` method
that does not exist.
"""

from __future__ import annotations

from tests.test_server.test_canonical_remember_route import _client


def test_recall_500_does_not_leak_exception_text(engine_with_mock_deps, monkeypatch) -> None:
    engine = engine_with_mock_deps

    def broken(*a, **k):
        raise FileNotFoundError(
            2, "No such file or directory",
            "/Users/alice/.superlocalmemory/laya/model.safetensors",
        )

    monkeypatch.setattr(engine, "recall", broken)
    with _client(engine) as client:
        r = client.get("/recall", params={"q": "anything"})
    assert r.status_code == 500
    assert "/Users/alice" not in r.text
    assert "safetensors" not in r.text


def test_components_500_does_not_leak_exception_text(engine_with_mock_deps, monkeypatch) -> None:
    from superlocalmemory.core import component_registry

    def broken(cfg):
        raise RuntimeError("/Users/alice/secret-internal-path blew up")

    monkeypatch.setattr(component_registry, "snapshot", broken)
    with _client(engine_with_mock_deps) as client:
        r = client.get("/api/v3/components")
    assert r.status_code == 500
    assert "/Users/alice" not in r.text


def test_list_works_instead_of_always_500ing(engine_with_mock_deps) -> None:
    """``engine.list_facts`` does not exist on MemoryEngine; /list must use
    the same canonical read every other surface uses instead of 500ing."""
    engine = engine_with_mock_deps
    from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
    mid = engine._db.store_memory(MemoryRecord(profile_id=engine.profile_id, content="s"))
    engine._db.store_fact(AtomicFact(
        profile_id=engine.profile_id, memory_id=mid, content="Never push to main.",
        fact_type=FactType.SEMANTIC, memory_kind="rule", memory_kind_source="user",
    ))
    with _client(engine) as client:
        r = client.get("/list")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["results"][0]["content"] == "Never push to main."


def test_list_kind_filter(engine_with_mock_deps) -> None:
    """L3-20: /list takes the same ``kind`` filter MCP's list_recent and the
    CLI's ``slm list`` already take."""
    engine = engine_with_mock_deps
    from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

    def _save(content, kind):
        mid = engine._db.store_memory(MemoryRecord(profile_id=engine.profile_id, content="s"))
        engine._db.store_fact(AtomicFact(
            profile_id=engine.profile_id, memory_id=mid, content=content,
            fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source="user",
        ))

    _save("rule one", "rule")
    _save("a decision", "decision")
    with _client(engine) as client:
        r = client.get("/list", params={"kind": "rule"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == 1
        assert body["results"][0]["content"] == "rule one"
        assert body["results"][0]["memory_kind"] == "rule"

        bad = client.get("/list", params={"kind": "not-a-real-kind"})
        assert bad.status_code == 422
        assert bad.json()["detail"]["code"] == "INVALID_KIND"
