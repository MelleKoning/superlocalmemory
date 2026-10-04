# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The "Also use Jev to reorder results" toggle: its consent, its gates, and
every way of turning it off.

It sends more memories than the answer check alone, so it has its own consent,
and anything that turns the online check off — or changes who receives the
text — turns it off too, persistently, so it never resumes unasked.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core import engine_wiring, judge_selection
from superlocalmemory.server.routes import answer_check

# Fixtures and fakes of the answer-check API suite (imported, not duplicated).
from tests.test_server.test_answer_check_api import (  # noqa: F401 — fixtures
    FAKE_KEY,
    _FakeKeyStore,
    _reset_fake_key_store,
    _retrieval,
    _set_retrieval,
    call_order,
    client,
    daemon_client,
)

URL = "/api/v3/answer-check/jev/rerank"
_UI = Path(answer_check.__file__).resolve().parents[2] / "ui" / "js"
CONSENT_TEMPLATE = (
    "When on, each recall sends your question and its top {k} memories to "
    "{provider} to choose the best order. Don't use this for client, "
    "confidential or personal material."
)


@pytest.fixture()
def attached(monkeypatch):
    """What the live judge would be rebuilt with on every attach."""
    seen: list[tuple[str, int]] = []

    def _attach(retrieval_engine, retrieval_config):
        mode = retrieval_config.sufficiency_judge
        k = judge_selection.jev_rerank_k(retrieval_config)
        seen.append((mode, k))
        # The status reports what runs, so put the judge this config builds
        # on the engine (the hosted check only with its consent).
        runs_jev = mode == "jev" and retrieval_config.sufficiency_jev_consent is True
        retrieval_engine._sufficiency_judge = (
            SimpleNamespace(backend="jev", rerank_k=k, shutdown=lambda: None)
            if runs_jev else None)
        return judge_selection.backend_of(retrieval_engine._sufficiency_judge)

    monkeypatch.setattr(engine_wiring, "attach_sufficiency_judge", _attach, raising=False)
    return seen


def _jev_on(tmp_path, **extra) -> None:
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY
    _set_retrieval(tmp_path, sufficiency_judge="jev", sufficiency_jev_consent=True,
                   sufficiency_jev_provider="typesafe", **extra)


def _reordering_on(tmp_path) -> None:
    _jev_on(tmp_path, sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True)


def _flags(tmp_path) -> tuple:
    r = _retrieval(tmp_path)
    return (r.get("sufficiency_jev_rerank", False), r.get("sufficiency_jev_rerank_consent", False))


def _enable(client, provider="typesafe", accepted=True):  # noqa: F811
    return client.post(URL, json={"enabled": True, "accepted": accepted, "provider": provider})


# ---------------------------------------------------------------------------
# turning it on
# ---------------------------------------------------------------------------

class TestTurningItOn:
    def test_on_with_everything_in_place(self, client, tmp_path, attached):  # noqa: F811
        _jev_on(tmp_path)
        response = _enable(client)
        assert response.status_code == 200, response.text
        assert _flags(tmp_path) == (True, True)
        assert attached[-1] == ("jev", 20), "the live judge is rebuilt to reorder"
        body = response.json()
        assert body["jev"]["rerank"] == {"enabled": True, "active": True, "k": 20}
        assert "reorder" in body["message"].lower()

    def test_the_notice_must_be_accepted(self, client, tmp_path, attached):  # noqa: F811
        _jev_on(tmp_path)
        response = _enable(client, accepted=False)
        assert response.status_code == 400
        assert "accept" in response.json()["error"].lower()
        assert _flags(tmp_path) == (False, False)
        assert attached == []

    def test_the_online_check_must_be_on(self, client, tmp_path):  # noqa: F811
        _jev_on(tmp_path)
        _set_retrieval(tmp_path, sufficiency_judge="off")
        assert _enable(client).status_code == 400
        assert _flags(tmp_path) == (False, False)

    def test_its_consent_must_be_given(self, client, tmp_path):  # noqa: F811
        _jev_on(tmp_path)
        _set_retrieval(tmp_path, sufficiency_jev_consent=False)
        assert _enable(client).status_code == 400
        assert _flags(tmp_path) == (False, False)

    def test_a_key_must_be_saved(self, client, tmp_path):  # noqa: F811
        _jev_on(tmp_path)
        _FakeKeyStore._by_test.clear()
        response = _enable(client)
        assert response.status_code == 400
        assert "key" in response.json()["error"].lower()
        assert _flags(tmp_path) == (False, False)

    def test_the_provider_named_in_the_notice_must_be_the_one_in_use(
        self, client, tmp_path,  # noqa: F811
    ):
        _jev_on(tmp_path)
        response = _enable(client, provider="openrouter")
        assert response.status_code == 409
        assert _flags(tmp_path) == (False, False)

    def test_a_change_that_lands_first_wins(self, client, tmp_path, monkeypatch):  # noqa: F811
        """The checks are repeated under the config lock: if Jev was switched
        off after they passed, this request must not turn reordering on."""
        _jev_on(tmp_path)
        monkeypatch.setattr(answer_check, "_rerank_refusal", lambda retrieval, body: None)
        _set_retrieval(tmp_path, sufficiency_judge="off")
        response = _enable(client)
        assert response.status_code == 409
        assert _flags(tmp_path) == (False, False)

    @pytest.mark.parametrize("body", [
        {"enabled": "true", "accepted": True, "provider": "typesafe"},
        {"enabled": True, "accepted": 1, "provider": "typesafe"},
        {"enabled": True, "accepted": True, "provider": "elsewhere"},
        {"enabled": True, "accepted": True, "provider": "typesafe", "k": 30},
        {"accepted": True, "provider": "typesafe"},
    ])
    def test_the_request_is_validated_strictly(self, client, tmp_path, body):  # noqa: F811
        _jev_on(tmp_path)
        assert client.post(URL, json=body).status_code == 422
        assert _flags(tmp_path) == (False, False)

    def test_a_malformed_count_takes_the_count_the_notice_showed(
            self, client, tmp_path, attached):  # noqa: F811
        """A hand-edited count that is not a number keeps reordering off — until
        the person ticks the box next to a notice that names a count. That
        count is what they agreed to, so it is the one stored."""
        _jev_on(tmp_path, sufficiency_jev_rerank_k="lots")
        shown = client.get("/api/v3/answer-check").json()["jev"]["rerank"]["k"]
        response = _enable(client)
        assert response.status_code == 200, response.text
        assert _retrieval(tmp_path)["sufficiency_jev_rerank_k"] == shown
        assert response.json()["jev"]["rerank"]["active"] is True

    def test_the_status_reports_the_clamped_count(self, client, tmp_path):  # noqa: F811
        _reordering_on(tmp_path)
        _set_retrieval(tmp_path, sufficiency_jev_rerank_k=99)
        client.app.state.engine._retrieval_engine._sufficiency_judge = SimpleNamespace(
            backend="jev", rerank_k=30)
        assert client.get("/api/v3/answer-check").json()["jev"]["rerank"] == {
            "enabled": True, "active": True, "k": 30}

    def test_off_by_default_in_the_status(self, client):  # noqa: F811
        assert client.get("/api/v3/answer-check").json()["jev"]["rerank"] == {
            "enabled": False, "active": False, "k": 20}


# ---------------------------------------------------------------------------
# every way of turning it off
# ---------------------------------------------------------------------------

class TestEveryWayOfTurningItOff:
    def test_turning_the_toggle_off(self, client, tmp_path, attached):  # noqa: F811
        _reordering_on(tmp_path)
        response = client.post(URL, json={"enabled": False, "accepted": False,
                                          "provider": "typesafe"})
        assert response.status_code == 200, response.text
        assert _flags(tmp_path) == (False, False)
        assert attached[-1] == ("jev", 0)
        assert response.json()["jev"]["rerank"]["enabled"] is False

    def test_turning_the_toggle_off_needs_no_gate(self, client, tmp_path):  # noqa: F811
        """Off is always allowed — even with the key already gone."""
        _set_retrieval(tmp_path, sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True)
        response = client.post(URL, json={"enabled": False, "accepted": False,
                                          "provider": "openrouter"})
        assert response.status_code == 200
        assert _flags(tmp_path) == (False, False)

    @pytest.mark.parametrize("mode", ["off", "laya"])
    def test_switching_the_answer_check_away_from_jev(
        self, client, tmp_path, monkeypatch, attached, mode,  # noqa: F811
    ):
        from superlocalmemory.core import laya_runtime

        ready = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_READY)
        monkeypatch.setattr(laya_runtime, "detect", lambda cfg=None: ready)
        _reordering_on(tmp_path)
        response = client.post("/api/v3/answer-check/mode", json={"mode": mode})
        assert response.status_code == 200
        # Reordering is part of Jev: nothing is sent for it while another option
        # is chosen (k == 0 in every rebuild) — and the choice is remembered,
        # said plainly, not silently reset.
        assert all(k == 0 for _, k in attached)
        assert _flags(tmp_path) == (True, True)
        rerank = response.json()["jev"]["rerank"]
        assert rerank["enabled"] is True and rerank["active"] is False
        assert "Reordering with Jev is off while it isn't chosen" in response.json()["message"]

    def test_choosing_jev_again_restores_the_remembered_choice(
        self, client, tmp_path, monkeypatch, attached,  # noqa: F811
    ):
        from superlocalmemory.core import laya_runtime

        ready = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_READY)
        monkeypatch.setattr(laya_runtime, "detect", lambda cfg=None: ready)
        _reordering_on(tmp_path)
        assert client.post("/api/v3/answer-check/mode", json={"mode": "laya"}).status_code == 200
        assert attached[-1][1] == 0
        assert client.post("/api/v3/answer-check/mode", json={"mode": "jev"}).status_code == 200
        assert attached[-1] == ("jev", 20)

    def test_choosing_jev_again_keeps_it(self, client, tmp_path):  # noqa: F811
        _reordering_on(tmp_path)
        assert client.post("/api/v3/answer-check/mode", json={"mode": "jev"}).status_code == 200
        assert _flags(tmp_path) == (True, True)

    def test_removing_the_key(self, client, tmp_path, attached):  # noqa: F811
        _reordering_on(tmp_path)
        response = client.request("DELETE", "/api/v3/answer-check/jev/key",
                                  json={"provider": "typesafe"})
        assert response.status_code == 200
        assert _flags(tmp_path) == (False, False)
        assert attached[-1] == ("off", 0)

    def test_withdrawing_the_answer_check_consent(self, client, tmp_path, attached):  # noqa: F811
        _reordering_on(tmp_path)
        response = client.post("/api/v3/answer-check/jev/consent",
                               json={"accepted": False, "provider": "typesafe"})
        assert response.status_code == 200
        assert _flags(tmp_path) == (False, False)
        assert attached[-1] == ("off", 0)

    @pytest.mark.parametrize("action", ["delete_key", "withdraw_consent"])
    def test_a_stale_setting_is_cleared_even_with_the_online_check_off(
        self, client, tmp_path, action,  # noqa: F811
    ):
        """Left on by hand with Jev off, it must not resume when Jev comes back."""
        _reordering_on(tmp_path)
        _set_retrieval(tmp_path, sufficiency_judge="off")
        if action == "delete_key":
            client.request("DELETE", "/api/v3/answer-check/jev/key", json={"provider": "typesafe"})
        else:
            client.post("/api/v3/answer-check/jev/consent",
                        json={"accepted": False, "provider": "typesafe"})
        assert _flags(tmp_path) == (False, False)

    def test_changing_the_provider_withdraws_it_and_rebuilds_the_live_check(
        self, client, tmp_path, attached,  # noqa: F811
    ):
        """The consent named one provider; the text must not go to another."""
        _reordering_on(tmp_path)
        response = client.post("/api/v3/answer-check/jev/consent",
                               json={"accepted": True, "provider": "openrouter"})
        assert response.status_code == 200
        assert _flags(tmp_path) == (False, False)
        assert attached[-1] == ("jev", 0)

    def test_reaccepting_the_same_provider_keeps_it(self, client, tmp_path):  # noqa: F811
        _reordering_on(tmp_path)
        client.post("/api/v3/answer-check/jev/consent",
                    json={"accepted": True, "provider": "typesafe"})
        assert _flags(tmp_path) == (True, True)


# ---------------------------------------------------------------------------
# the key never leaks; auth is the same as every other answer-check mutation
# ---------------------------------------------------------------------------

def test_the_key_is_never_in_a_response_or_a_log(client, tmp_path, caplog):  # noqa: F811
    caplog.set_level(logging.DEBUG)
    _jev_on(tmp_path)
    texts = [_enable(client).text,
             client.get("/api/v3/answer-check").text,
             client.post(URL, json={"enabled": False, "accepted": False,
                                    "provider": "typesafe"}).text]
    assert all(FAKE_KEY not in t for t in texts)
    assert FAKE_KEY not in caplog.text


def test_an_internal_failure_is_reported_plainly(client, tmp_path, monkeypatch, caplog):  # noqa: F811
    _jev_on(tmp_path)

    def _boom(change):
        raise OSError("disk full")

    monkeypatch.setattr(answer_check.support, "save", _boom)
    response = _enable(client)
    assert response.status_code == 500
    assert response.json() == {"error": "Internal server error"}
    assert FAKE_KEY not in response.text and FAKE_KEY not in caplog.text


def _viewer_token(tc, h) -> str:
    tc.post("/api/rbac/users",
            json={"username": "rr-viewer", "password": "password-1234", "role": "viewer"},
            headers=h)
    rbac = tc.app.state.rbac
    user_id = {u["username"]: u["user_id"] for u in rbac.list_users()}["rr-viewer"]
    return rbac.create_session(user_id)


@pytest.mark.parametrize("body", [
    {"enabled": True, "accepted": True, "provider": "typesafe"},
    {"enabled": False, "accepted": False, "provider": "typesafe"},
])
def test_a_viewer_cannot_change_it(daemon_client, body):  # noqa: F811
    tc, h = daemon_client
    token = _viewer_token(tc, h)
    response = tc.post(URL, json=body, headers={**h, "X-SLM-User-Session": token})
    assert response.status_code == 403, response.text


def test_an_uncredentialed_cross_origin_write_is_rejected(daemon_client):  # noqa: F811
    tc, _h = daemon_client
    response = tc.post(URL, json={"enabled": False, "accepted": False, "provider": "typesafe"},
                       headers={"Origin": "http://localhost:8417"})
    assert response.status_code == 403, response.text


def test_the_owner_clears_both_gates(daemon_client):  # noqa: F811
    tc, h = daemon_client
    response = tc.post(URL, json={"enabled": False, "accepted": False, "provider": "typesafe"},
                       headers=h)
    assert response.status_code not in (401, 403), response.text


# ---------------------------------------------------------------------------
# the words on the screen
# ---------------------------------------------------------------------------

def test_the_consent_text_is_exactly_the_approved_wording():
    source = (_UI / "answer-check.js").read_text(encoding="utf-8")
    assert f'RERANK_CONSENT_TEXT =\n    "{CONSENT_TEMPLATE}";' in source
    assert "Also use Jev to reorder results" in source


def test_the_dashboard_row_says_when_reordering_is_on():
    source = (_UI / "dashboard.js").read_text(encoding="utf-8")
    assert "reordering on" in source
