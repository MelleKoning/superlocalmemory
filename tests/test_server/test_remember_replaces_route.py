# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""POST /remember with ``replaces``: a bad id is refused (422) before anything is
saved; a good one retires the old memory after the new one is saved; a failed
mark is reported, never hidden; and recall then answers with the new memory."""

from __future__ import annotations

import pytest

from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from tests.test_server.test_canonical_remember_route import _client

OLD = "The release train for the platform team leaves on Tuesday at 14:00 UTC."
NEW = "The release train for the platform team leaves on Thursday at 09:00 UTC."


def _count(engine, table: str) -> int:
    return len(engine._db.execute(f"SELECT 1 FROM {table}"))


def _expired(engine, fact_id: str) -> dict:
    rows = engine._db.execute(
        "SELECT system_expired_at, invalidated_by, invalidation_reason "
        "FROM fact_temporal_validity WHERE fact_id=?", (fact_id,))
    return dict(rows[0]) if rows else {}


def _save(client, content: str, key: str, **extra):
    response = client.post("/remember", json={"content": content, "idempotency_key": key,
                                              **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _foreign_fact(engine, scope: str) -> str:
    db = engine._db
    db.execute("INSERT OR IGNORE INTO profiles(profile_id, name, description) "
               "VALUES ('other', 'other', 'test')")
    content = f"Their {scope} release train leaves on Monday."
    memory_id = db.store_memory(MemoryRecord(profile_id="other", content=content,
                                             scope=scope))
    fact_id = db.store_fact(AtomicFact(profile_id="other", memory_id=memory_id,
                                       content=content, fact_type=FactType.SEMANTIC,
                                       scope=scope))
    assert dict(db.execute("SELECT scope FROM atomic_facts WHERE fact_id=?",
                           (fact_id,))[0])["scope"] == scope
    return fact_id


def test_a_memory_replaced_by_the_id_remember_returned(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        old = _save(client, OLD, "replaces-old-1")
        [old_id] = old["fact_ids"]
        new = _save(client, NEW, "replaces-new-1", replaces=old_id)
    [new_id] = new["fact_ids"]
    assert new["replaced"]["ok"] is True, new["replaced"]
    assert new["replaced"]["fact_ids"] == [old_id]
    assert _expired(engine, old_id)["invalidated_by"] == new_id
    assert _expired(engine, new_id)["system_expired_at"] is None


def test_the_id_remember_returned_also_retires_facts_extracted_later(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        [old_id] = _save(client, OLD, "replaces-old-2")["fact_ids"]
        memory_id = dict(engine._db.execute(
            "SELECT memory_id FROM atomic_facts WHERE fact_id=?", (old_id,))[0])["memory_id"]
        extracted = engine._db.store_fact(AtomicFact(
            profile_id="default", memory_id=memory_id, fact_type=FactType.SEMANTIC,
            content="The platform release train departs Tuesdays."))
        new = _save(client, NEW, "replaces-new-2", replaces=old_id)
    assert sorted(new["replaced"]["fact_ids"]) == sorted([old_id, extracted])
    assert _expired(engine, extracted)["system_expired_at"]


def test_a_plain_remember_response_is_unchanged(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        body = _save(client, OLD, "replaces-legacy-1")
    assert "replaced" not in body


@pytest.mark.parametrize("value,code", [
    ("", "INVALID_REPLACES"),
    ("   ", "INVALID_REPLACES"),
    ("not an id", "INVALID_REPLACES"),
    ("0123456789abcdef", "REPLACES_NOT_FOUND"),
])
def test_a_bad_replaces_is_refused_before_anything_is_saved(engine_with_mock_deps,
                                                            value, code) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        response = client.post("/remember", json={"content": NEW, "replaces": value})
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == code
    assert _count(engine, "memories") == 0 and _count(engine, "atomic_facts") == 0


@pytest.mark.parametrize("value", [7, ["abc"], {"id": "x"}])
def test_a_replaces_of_the_wrong_type_is_refused(engine_with_mock_deps, value) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        response = client.post("/remember", json={"content": NEW, "replaces": value})
    assert response.status_code == 422, response.text
    assert _count(engine, "memories") == 0


def test_another_profiles_memory_is_refused_plainly(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    shown = _foreign_fact(engine, "global")
    private = _foreign_fact(engine, "personal")
    before = _count(engine, "memories")
    with _client(engine) as client:
        visible = client.post("/remember", json={"content": NEW, "replaces": shown,
                                                 "scope": "global"})
        hidden = client.post("/remember", json={"content": NEW, "replaces": private})
    assert visible.status_code == 422
    assert visible.json()["detail"]["code"] == "REPLACES_NOT_ALLOWED"
    assert "another profile" in visible.json()["detail"]["message"]
    assert hidden.status_code == 422
    assert hidden.json()["detail"]["code"] == "REPLACES_NOT_FOUND"
    assert _count(engine, "memories") == before
    assert _expired(engine, shown)["system_expired_at"] is None


def test_a_failed_mark_keeps_the_memory_and_says_so(engine_with_mock_deps, monkeypatch) -> None:
    import sqlite3

    from superlocalmemory.core import remember_replaces

    def broken(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")

    engine = engine_with_mock_deps
    with _client(engine) as client:
        [old_id] = _save(client, OLD, "replaces-old-3")["fact_ids"]
        monkeypatch.setattr(remember_replaces, "transition_on_connection", broken)
        new = _save(client, NEW, "replaces-new-3", replaces=old_id)
    assert new["ok"] is True and len(new["fact_ids"]) == 1
    assert new["replaced"]["ok"] is False and new["replaced"]["reason"]
    assert _expired(engine, old_id)["system_expired_at"] is None
    assert engine._db.execute("SELECT 1 FROM atomic_facts WHERE fact_id=?",
                              (new["fact_ids"][0],))


def test_replaying_the_request_does_not_mark_twice(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        [old_id] = _save(client, OLD, "replaces-old-4")["fact_ids"]
        first = _save(client, NEW, "replaces-new-4", replaces=old_id)
        stamped = _expired(engine, old_id)
        second = _save(client, NEW, "replaces-new-4", replaces=old_id)
    assert first["replaced"] == second["replaced"]
    assert first["operation_id"] == second["operation_id"]
    assert _count(engine, "correction_cases") == 1
    assert _expired(engine, old_id) == stamped


def test_replacing_needs_the_permission_to_correct(engine_with_mock_deps, monkeypatch) -> None:
    """REMEMBER is open to members; CORRECT (what review_correction needs) is not.
    A caller allowed to save but not to correct must not retire anything."""
    from types import SimpleNamespace

    from superlocalmemory.core import operation_policy_registry as registry
    from superlocalmemory.core.operation_request import OperationKind

    engine = engine_with_mock_deps
    real = registry._DEFAULT_REGISTRY.evaluate
    asked = []

    def members_only(kind, actor, mode):
        asked.append(kind)
        if kind is OperationKind.CORRECT:
            return SimpleNamespace(allowed=False, reason="member role")
        return real(kind, actor, mode)

    with _client(engine) as client:
        [old_id] = _save(client, OLD, "replaces-old-7")["fact_ids"]
        before = _count(engine, "memories")
        monkeypatch.setattr(registry._DEFAULT_REGISTRY, "evaluate", members_only)
        refused = client.post("/remember", json={"content": NEW, "replaces": old_id,
                                                 "idempotency_key": "replaces-new-7"})
        plain = client.post("/remember", json={"content": NEW,
                                               "idempotency_key": "replaces-new-8"})
    assert refused.status_code == 403, refused.text
    assert OperationKind.CORRECT in asked
    assert _expired(engine, old_id)["system_expired_at"] is None
    assert plain.status_code == 200
    assert _count(engine, "memories") == before + 1


def test_a_dashboard_writer_can_replace(engine_with_mock_deps) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    engine = engine_with_mock_deps
    with _client(engine) as client:
        [old_id] = _save(client, OLD, "replaces-old-5")["fact_ids"]
        client.headers.pop("X-SLM-Daemon-Capability")
        client.headers.pop("X-SLM-Target-Instance")
        response = client.post(
            "/remember", json={"content": NEW, "idempotency_key": "replaces-new-5",
                               "replaces": old_id},
            headers={"X-Install-Token": ensure_install_token()})
    assert response.status_code == 200, response.text
    assert response.json()["replaced"]["ok"] is True, response.json()["replaced"]


def test_recall_answers_with_the_new_memory(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    query = "When does the platform release train leave?"
    with _client(engine) as client:
        [old_id] = _save(client, OLD, "replaces-old-6")["fact_ids"]
        before = client.get("/recall", params={"q": query, "answer_check": "skip"}).json()
        assert old_id in [r["fact_id"] for r in before["results"]], before
        [new_id] = _save(client, NEW, "replaces-new-6", replaces=old_id)["fact_ids"]
        after = client.get("/recall", params={"q": query, "answer_check": "skip"}).json()
    ids = [r["fact_id"] for r in after["results"]]
    assert ids and ids[0] == new_id, after
    assert old_id not in ids
