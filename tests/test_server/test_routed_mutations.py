# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``replaces`` and correction review with a per-request ``profile_id``.

A client routed to its own profile can replace a memory there and undo it
there, without moving the daemon's active profile. Driven through the real
daemon routes, the real canonical writer and the real correction ledger: the
active profile is ``default`` and the client is routed to ``work``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_server.test_per_request_profile import _daemon

OLD = "The release train for the platform team leaves on Tuesday at 14:00 UTC."
NEW = "The release train for the platform team leaves on Thursday at 09:00 UTC."


@pytest.fixture
def routed(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps, profiles=("work", "ops")) as pair:
        yield pair


def _save(client, content: str, key: str, profile: str = "work", **extra) -> dict:
    response = client.post("/remember", json={"content": content, "profile_id": profile,
                                              "idempotency_key": key, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _expiry(engine, fact_id: str) -> dict:
    rows = engine._db.execute(
        "SELECT system_expired_at, invalidated_by, invalidation_reason "
        "FROM fact_temporal_validity WHERE fact_id=?", (fact_id,))
    return dict(rows[0]) if rows else {}


def _cases(engine, profile_id: str) -> list[dict]:
    rows = engine._db.execute(
        "SELECT case_id, profile_id, predecessor_fact_id, successor_fact_id, "
        "reason_code, status, version FROM correction_cases WHERE profile_id=?",
        (profile_id,))
    return [dict(r) for r in rows]


def _count(engine, table: str) -> int:
    return len(engine._db.execute(f"SELECT 1 FROM {table}"))


def _replace(client, tag: str) -> tuple[str, str, dict]:
    [old_id] = _save(client, OLD, f"{tag}-old")["fact_ids"]
    body = _save(client, NEW, f"{tag}-new", replaces=old_id)
    [new_id] = body["fact_ids"]
    return old_id, new_id, body


class _Wrap:
    """Run ``before`` between the save and the mark of a routed replace."""

    def __init__(self, monkeypatch, before) -> None:
        from superlocalmemory.core import remember_replaces

        real = remember_replaces.replace_after_save

        def wrapped(*args, **kwargs):
            before()
            return real(*args, **kwargs)

        monkeypatch.setattr(remember_replaces, "replace_after_save", wrapped)


# ---------------------------------------------------------------------------
# A routed replace takes effect in the routed profile, and only there
# ---------------------------------------------------------------------------

def test_a_routed_replace_retires_the_old_memory_in_that_profile(routed) -> None:
    client, app = routed
    engine = app.state.engine
    old_id, new_id, body = _replace(client, "i1")

    assert body["profile"] == "work"
    assert body["replaced"]["ok"] is True, body["replaced"]
    assert body["replaced"]["fact_ids"] == [old_id]
    assert _expiry(engine, old_id)["invalidated_by"] == new_id
    assert _expiry(engine, old_id)["system_expired_at"]
    assert _expiry(engine, new_id).get("system_expired_at") is None
    [case] = _cases(engine, "work")
    assert (case["predecessor_fact_id"], case["successor_fact_id"]) == (old_id, new_id)
    assert (case["reason_code"], case["status"]) == ("replaced_by_caller", "applied")


def test_a_routed_replace_leaves_the_active_profile_alone(routed) -> None:
    client, app = routed
    engine = app.state.engine
    before = client.get("/status").json()
    _replace(client, "i1b")
    after = client.get("/status").json()

    assert engine._profile_id == "default"
    assert app.state.canonical_remember_runtime._profile_id == "default"
    assert (after["profile"], after["profile_generation"]) == (
        before["profile"], before["profile_generation"])
    assert _cases(engine, "default") == [] and _cases(engine, "ops") == []


def test_the_undo_hint_names_the_routed_profile(routed) -> None:
    client, _app = routed
    _old, _new, body = _replace(client, "i1c")
    assert "profile_id='work'" in body["replaced"]["undo"], body["replaced"]


def test_routed_recall_answers_with_the_new_memory(routed) -> None:
    client, _app = routed
    query = "When does the platform release train leave?"
    old_id, new_id, _body = _replace(client, "i2")
    after = client.get("/recall", params={"q": query, "profile_id": "work",
                                          "answer_check": "skip"}).json()
    ids = [r["fact_id"] for r in after["results"]]
    assert ids and ids[0] == new_id, after
    assert old_id not in ids


def test_a_repeated_routed_replace_is_answered_from_the_first(routed) -> None:
    client, app = routed
    engine = app.state.engine
    [old_id] = _save(client, OLD, "i5-old")["fact_ids"]
    first = _save(client, NEW, "i5-new", replaces=old_id)
    stamped = _expiry(engine, old_id)
    second = _save(client, NEW, "i5-new", replaces=old_id)

    assert first["replaced"]["ok"] is True, first["replaced"]
    assert first["replaced"] == second["replaced"]
    assert len(_cases(engine, "work")) == 1
    assert _expiry(engine, old_id) == stamped


def test_a_profile_switch_between_save_and_mark_does_not_misdirect_it(
        routed, monkeypatch) -> None:
    """The routed mark names its profile, so the daemon's binding is irrelevant."""
    client, app = routed
    engine = app.state.engine
    runtime = app.state.canonical_remember_runtime
    [old_id] = _save(client, OLD, "i4-old")["fact_ids"]
    rebound = SimpleNamespace(_db=engine._db, _profile_id="ops", _config=engine._config)
    _Wrap(monkeypatch, lambda: runtime.rebind_engine(rebound))
    body = _save(client, NEW, "i4-new", replaces=old_id)

    assert runtime._profile_id == "ops", "the rebind must really have happened"
    assert body["replaced"]["ok"] is True, body["replaced"]
    assert _expiry(engine, old_id)["system_expired_at"]
    assert _cases(engine, "ops") == []


# ---------------------------------------------------------------------------
# Refusals stay refusals
# ---------------------------------------------------------------------------

def test_a_routed_write_cannot_replace_another_profiles_private_memory(routed) -> None:
    client, app = routed
    engine = app.state.engine
    [old_id] = _save(client, OLD, "i6-old", profile="work")["fact_ids"]
    before = _count(engine, "memories")
    refused = client.post("/remember", json={"content": NEW, "profile_id": "ops",
                                              "replaces": old_id,
                                              "idempotency_key": "i6-new"})

    assert refused.status_code == 422, refused.text
    # Same answer as an id that does not exist: nothing about 'work' leaks.
    assert refused.json()["detail"]["code"] == "REPLACES_NOT_FOUND"
    assert _count(engine, "memories") == before
    assert _expiry(engine, old_id).get("system_expired_at") is None


def test_a_profile_deleted_before_the_mark_is_reported_without_a_retry(
        routed, monkeypatch) -> None:
    client, app = routed
    engine = app.state.engine
    [old_id] = _save(client, OLD, "i7-old")["fact_ids"]

    def drop_profile() -> None:
        with engine._db.raw_connection() as conn:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("DELETE FROM profiles WHERE profile_id='work'")
            conn.commit()

    _Wrap(monkeypatch, drop_profile)
    body = _save(client, NEW, "i7-new", replaces=old_id)

    assert body["replaced"]["ok"] is False, body["replaced"]
    assert "no longer exists" in body["replaced"]["reason"]
    assert "Repeat" not in body["replaced"]["reason"]
    assert _cases(engine, "work") == []


# ---------------------------------------------------------------------------
# Correction review is routed the same way, so a routed replace can be undone
# ---------------------------------------------------------------------------

def _review(client, case: dict, action: str, profile: str | None = "work"):
    body = {"expected_version": case["version"]}
    if profile is not None:
        body["profile_id"] = profile
    return client.post(f"/api/corrections/{case['case_id']}/{action}", json=body)


def test_a_routed_replacement_is_undone_in_its_profile(routed) -> None:
    client, app = routed
    engine = app.state.engine
    old_id, new_id, body = _replace(client, "i3")
    [case] = body["replaced"]["cases"]

    undone = _review(client, case, "rollback")

    assert undone.status_code == 200, undone.text
    assert undone.json()["correction_case"]["status"] == "rolled_back"
    assert _expiry(engine, old_id).get("system_expired_at") is None
    assert _expiry(engine, new_id).get("system_expired_at") is None
    assert engine._profile_id == "default"


def test_without_profile_id_review_still_means_the_active_profile(routed) -> None:
    client, app = routed
    _old, _new, body = _replace(client, "i3b")
    [case] = body["replaced"]["cases"]
    legacy = _review(client, case, "rollback", profile=None)
    assert legacy.status_code == 404, legacy.text
    assert _cases(app.state.engine, "work")[0]["status"] == "applied"


def test_routed_listing_and_lookup_see_the_routed_profile(routed) -> None:
    client, _app = routed
    _old, _new, body = _replace(client, "i3c")
    [case] = body["replaced"]["cases"]

    listed = client.get("/api/corrections", params={"profile_id": "work"})
    one = client.get(f"/api/corrections/{case['case_id']}", params={"profile_id": "work"})
    legacy = client.get("/api/corrections")

    assert listed.status_code == 200, listed.text
    assert [c["case_id"] for c in listed.json()["corrections"]] == [case["case_id"]]
    assert one.status_code == 200 and one.json()["correction"]["profile_id"] == "work"
    assert legacy.json()["corrections"] == []


@pytest.mark.parametrize("call", ["review", "list", "get"])
def test_review_routed_to_an_unknown_profile_is_404(routed, call) -> None:
    client, _app = routed
    if call == "review":
        response = client.post("/api/corrections/abc/rollback",
                               json={"expected_version": 1, "profile_id": "ghost"})
    elif call == "list":
        response = client.get("/api/corrections", params={"profile_id": "ghost"})
    else:
        response = client.get("/api/corrections/abc", params={"profile_id": "ghost"})
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "unknown_profile"


def test_a_review_that_found_nothing_does_not_spend_its_retry_key(routed) -> None:
    """A miss is not a committed outcome: once the case exists, the same retry
    key must reach it instead of replaying the earlier "not found"."""
    client, _app = routed
    _old, _new, body = _replace(client, "i9")
    [case] = body["replaced"]["cases"]
    key = {"X-Idempotency-Key": "i9-review-key"}
    payload = {"expected_version": case["version"], "profile_id": "work"}

    missed = client.post("/api/corrections/0000000000000000/rollback", json=payload,
                         headers=key)
    hit = client.post(f"/api/corrections/{case['case_id']}/rollback", json=payload,
                      headers=key)

    assert missed.status_code == 404, missed.text
    assert hit.status_code == 200, hit.text


def test_a_missing_target_is_a_404_not_a_conflict_by_default() -> None:
    """Any route that maps writer errors generically must say "not found"."""
    from superlocalmemory.core.remember_runtime import (
        CanonicalMutationConflict,
        CaseNotInProfile,
        UnknownMutationProfile,
    )
    from superlocalmemory.server.routes.memories import _canonical_mutation_error

    for exc in (UnknownMutationProfile("x"), CaseNotInProfile("x")):
        assert not isinstance(exc, CanonicalMutationConflict)
        assert _canonical_mutation_error(exc, "x").status_code == 404


def test_a_profile_deleted_during_a_routed_review_is_404_unknown_profile(
        routed, monkeypatch) -> None:
    """The route saw the profile; the writer, a moment later, does not."""
    client, app = routed
    engine = app.state.engine
    runtime = app.state.canonical_remember_runtime
    _old, _new, body = _replace(client, "i10")
    [case] = body["replaced"]["cases"]
    real = runtime.transition_correction

    def drop_then_review(*args, **kwargs):
        with engine._db.raw_connection() as conn:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("DELETE FROM profiles WHERE profile_id='work'")
            conn.commit()
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "transition_correction", drop_then_review)
    response = _review(client, case, "rollback")

    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "unknown_profile"
    assert _cases(engine, "work")[0]["status"] == "applied"


def test_a_profile_id_in_a_route_that_does_not_route_is_ignored(routed) -> None:
    """Only replace and review route. A profile_id slipped into another
    mutation's body changes nothing: the active profile is acted on, and it
    does not own the routed profile's memory."""
    client, app = routed
    engine = app.state.engine
    [fact_id] = _save(client, OLD, "i11-old")["fact_ids"]

    response = client.patch(f"/api/memories/{fact_id}/scope",
                            json={"scope": "global", "profile_id": "work"})

    assert response.status_code == 404, response.text
    rows = engine._db.execute("SELECT scope FROM atomic_facts WHERE fact_id=?", (fact_id,))
    assert dict(rows[0])["scope"] == "personal"


def test_review_rejects_a_profile_id_that_is_not_text(routed) -> None:
    client, _app = routed
    response = client.post("/api/corrections/abc/rollback",
                           json={"expected_version": 1, "profile_id": 7})
    assert response.status_code == 422, response.text


def test_a_waiting_candidate_in_the_routed_profile_can_be_reviewed_there(routed) -> None:
    """A person's edit waiting for review blocks the replace; the refusal names
    the profile, and the routed listing and review are what clear it. (A
    candidate SLM proposed by itself no longer blocks: 4.1.22, overtaken.)"""
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
    )

    client, app = routed
    engine = app.state.engine
    [old_id] = _save(client, OLD, "i8-old")["fact_ids"]
    [other] = _save(client, "The platform release train was moved once already.",
                     "i8-other")["fact_ids"]
    actor = CorrectionActor(actor_id="a-person", actor_kind="host_authenticated",
                            trust_tier="trusted")
    with engine._db.raw_connection() as conn:
        propose_on_connection(
            conn, case_id="i8waitingcase000", profile_id="work", scope="personal",
            predecessor_fact_id=old_id, successor_fact_id=other,
            reason_code="direct_content_correction", actor=actor,
            idempotency_key="i8-waiting", is_profile_active=lambda p: p == "work",
            is_actor_trusted=lambda a: a == actor)
        conn.commit()

    blocked = _save(client, NEW, "i8-new", replaces=old_id)
    assert blocked["replaced"]["ok"] is False, blocked["replaced"]
    assert "profile 'work'" in blocked["replaced"]["reason"]

    listed = client.get("/api/corrections", params={"profile_id": "work"}).json()
    [waiting] = [c for c in listed["corrections"] if c["status"] == "proposed"]
    assert _review(client, waiting, "reject").status_code == 200

    retried = _save(client, NEW, "i8-new", replaces=old_id)
    assert retried["replaced"]["ok"] is True, retried["replaced"]
