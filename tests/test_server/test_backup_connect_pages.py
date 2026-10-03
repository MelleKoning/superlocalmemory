# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The connect pages show the recovery key the one time it is created."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from superlocalmemory.infra import backup_keys as bk
from superlocalmemory.server.routes import backup

_KEY_TEXT = bk.format_recovery_key(bytes(range(32)))
_CREATED = {"encryption": {"enabled": True, "recovery_key": _KEY_TEXT}}
_EXISTING = {"encryption": {"enabled": True, "key_id": "00"}}


def _request() -> Request:
    return Request({
        "type": "http", "method": "GET", "path": "/", "headers": [],
        "client": ("127.0.0.1", 1), "server": ("localhost", 8765),
        "scheme": "http", "app": object(),
    })


@pytest.fixture
def callbacks(monkeypatch):
    monkeypatch.setattr(backup, "_consume_oauth_state", lambda *a, **k: True)
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: SimpleNamespace(json=lambda: {"access_token": "t"}),
    )
    import superlocalmemory.infra.cloud_backup as cb

    monkeypatch.setattr(cb, "_get_credential", lambda name: "configured")


@pytest.mark.parametrize("result,shown", [(_CREATED, True), (_EXISTING, False)])
def test_github_callback_page_shows_the_key_only_when_created(callbacks, monkeypatch, result, shown) -> None:
    monkeypatch.setattr(backup, "connect_github", lambda *a: {"repo": "u/r", **result})
    page = backup.github_oauth_callback(_request(), code="c", state="s").body.decode()
    assert (_KEY_TEXT in page) is shown
    assert "recovery key" in page


@pytest.mark.parametrize("result,shown", [(_CREATED, True), (_EXISTING, False)])
def test_google_callback_page_shows_the_key_only_when_created(callbacks, monkeypatch, result, shown) -> None:
    monkeypatch.setattr(backup, "connect_google_drive", lambda *a: {"email": "e@x", **result})
    page = backup.google_oauth_callback(_request(), code="c", state="s").body.decode()
    assert (_KEY_TEXT in page) is shown
    assert "recovery key" in page


def test_token_form_renders_the_recovery_key_as_text(monkeypatch) -> None:
    import superlocalmemory.infra.cloud_backup as cb

    monkeypatch.setattr(backup, "_require_oauth_start", lambda request: None)
    monkeypatch.setattr(backup, "CLOUD_AVAILABLE", True)
    monkeypatch.setattr(cb, "_get_credential", lambda name: None)
    page = backup.github_oauth_start(_request()).body.decode()
    assert "data.encryption" in page and "recovery_key" in page
    assert "keyText.textContent = data.encryption.recovery_key" in page
    assert "innerHTML" not in page
