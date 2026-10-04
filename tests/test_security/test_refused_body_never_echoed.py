# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A request body the daemon refuses is never quoted back in the answer.

Found against a running daemon: ``POST /api/v3/answer-check/jev/key`` with a
key one character over the limit answered 422 with the whole key in the
``input`` field, before any credential check ran. Extra fields came back the
same way. The fix is app-wide, so a password or API key sent to any other
route is covered too.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

_MARKER = "MARKERk3y"


def _client() -> TestClient:
    from superlocalmemory.server.unified_daemon import create_app

    return TestClient(create_app(), base_url="http://127.0.0.1:8765")


def _post(client: TestClient, body: dict):
    return client.post("/api/v3/answer-check/jev/key", json=body,
                       headers={"Host": "127.0.0.1:8765"})


def test_an_over_long_key_is_not_quoted_back() -> None:
    resp = _post(_client(), {"provider": "typesafe", "key": _MARKER + "x" * 4096})
    assert resp.status_code == 422
    assert _MARKER not in resp.text
    error = resp.json()["detail"][0]
    assert error["loc"] == ["body", "key"]
    assert error["type"] == "string_too_long"


def test_an_extra_field_is_not_quoted_back() -> None:
    resp = _post(_client(), {"provider": "typesafe", "key": "k" * 40, "secret": _MARKER})
    assert resp.status_code == 422
    assert _MARKER not in resp.text
    assert resp.json()["detail"][0]["loc"] == ["body", "secret"]


def test_the_error_still_says_what_to_fix() -> None:
    resp = _post(_client(), {"provider": "nope", "key": "k" * 40})
    assert resp.status_code == 422
    error = resp.json()["detail"][0]
    assert error["loc"] == ["body", "provider"]
    assert "pattern" in error["msg"]
    assert "input" not in error
