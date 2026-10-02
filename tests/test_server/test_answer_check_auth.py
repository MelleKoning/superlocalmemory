# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Changing the answer check needs the dashboard's own credential, even locally.

These settings decide where memory text goes (a hosted provider and account)
and what gets billed. Any process on this machine reaching the daemon without a
credential used to count as the owner and could store its own key, accept the
notice and turn the online check on — sending every recall's memories to an
account it controls. Every change under /api/v3/answer-check now needs one of
the credentials the product already issues: the install token the dashboard
sends on every change, the daemon capability, or an API key.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import security_primitives
from superlocalmemory.server.routes import answer_check

from tests.test_server.test_answer_check_api import (  # noqa: F401 — fixtures
    FAKE_KEY,
    _FakeKeyStore,
    _reset_fake_key_store,
    call_order,
    client,
)

TOKEN = "install-token-for-this-test-0123456789abcdef"

MUTATIONS = (
    ("POST", "/api/v3/answer-check/mode", {"mode": "off"}),
    ("POST", "/api/v3/answer-check/laya/setup", None),
    ("POST", "/api/v3/answer-check/laya/adopt", {"python": "/usr/bin/python3"}),
    ("POST", "/api/v3/answer-check/laya/remove", None),
    ("POST", "/api/v3/answer-check/jev/key", {"provider": "typesafe", "key": FAKE_KEY}),
    ("DELETE", "/api/v3/answer-check/jev/key", {"provider": "typesafe"}),
    ("POST", "/api/v3/answer-check/jev/test", {"provider": "typesafe"}),
    ("POST", "/api/v3/answer-check/jev/consent", {"accepted": True, "provider": "typesafe"}),
    ("POST", "/api/v3/answer-check/jev/rerank",
     {"enabled": False, "accepted": False, "provider": "typesafe"}),
)


#: The real gate, captured before any fixture replaces it.
_REAL_GATE = getattr(answer_check, "_require_credential", None)


@pytest.fixture()
def strict(client, monkeypatch, tmp_path):  # noqa: F811
    """The suite's client with the real credential gate put back (the role
    check stays stubbed: it has its own tests)."""
    if _REAL_GATE is not None:
        monkeypatch.setattr(answer_check, "_require_credential", _REAL_GATE)
    else:
        monkeypatch.delattr(answer_check, "_require_credential", raising=False)
    token_file = tmp_path / ".install_token"
    token_file.write_text(TOKEN)
    monkeypatch.setattr(security_primitives, "_install_token_path", lambda: token_file)
    return client


@pytest.mark.parametrize("method,path,body", MUTATIONS)
def test_a_change_without_a_credential_is_refused(strict, method, path, body):
    response = strict.request(method, path, json=body)
    assert response.status_code == 403, response.text


@pytest.mark.parametrize("method,path,body", MUTATIONS)
def test_a_wrong_token_is_refused(strict, method, path, body):
    response = strict.request(method, path, json=body,
                              headers={"X-Install-Token": "not-the-token"})
    assert response.status_code == 403, response.text


def test_the_token_the_dashboard_sends_is_accepted(strict, tmp_path):
    response = strict.post("/api/v3/answer-check/mode", json={"mode": "off"},
                           headers={"X-Install-Token": TOKEN})
    assert response.status_code == 200, response.text


def test_nothing_changes_when_a_change_is_refused(strict, tmp_path):
    _FakeKeyStore._by_test.clear()
    strict.post("/api/v3/answer-check/jev/key", json={"provider": "typesafe", "key": FAKE_KEY})
    strict.post("/api/v3/answer-check/jev/consent",
                json={"accepted": True, "provider": "typesafe"})
    assert _FakeKeyStore._by_test == {}
    assert not (tmp_path / "answer_check.json").exists()


def test_reading_the_status_needs_no_credential(strict):
    assert strict.get("/api/v3/answer-check").status_code == 200


def test_the_module_no_longer_says_loopback_is_enough():
    doc = answer_check.__doc__ or ""
    assert "even from this machine" in doc
