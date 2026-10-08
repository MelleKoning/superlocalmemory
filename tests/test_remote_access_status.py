"""Web access status for a completed connection: renews, ending soon, ended, sign in again."""
import json
import time
from dataclasses import replace

import pytest

from tests.test_remote_connection_runtime import enrolled_runtime

DAY_MS = 86_400_000


def completed_status(tmp_path, *, remaining_ms, transport=None, verified=False):
    runtime, row = enrolled_runtime(tmp_path)
    runtime.store.save(
        replace(row, completed=True, expires_at_ms=int(time.time() * 1000) + remaining_ms)
    )
    if verified:
        runtime._verified.add(row.connection_id)
    if transport:
        runtime._states[row.connection_id] = transport
    status = runtime.service.status(row.owner, row.profile)
    return status["connections"][0], status


@pytest.mark.parametrize(
    ("remaining_days", "expected"),
    [(29, "renews_automatically"), (7.5, "renews_automatically"), (6.9, "ending_soon"),
     (0.01, "ending_soon"), (-0.01, "ended"), (-40, "ended")],
)
def test_completed_connection_reports_access_state(tmp_path, remaining_days, expected):
    connection, _ = completed_status(
        tmp_path, remaining_ms=int(remaining_days * DAY_MS), transport="transport_ready"
    )
    assert connection["access_state"] == expected
    assert isinstance(connection["access_expires_at_ms"], int)


def test_access_expiry_follows_the_renewed_credential(tmp_path):
    runtime, row = enrolled_runtime(tmp_path)
    now = int(time.time() * 1000)
    runtime.store.save(replace(row, completed=True, expires_at_ms=now + 3 * DAY_MS))
    first = runtime.service.status(row.owner, row.profile)["connections"][0]
    assert first["access_state"] == "ending_soon"
    runtime.store.save(replace(row, completed=True, expires_at_ms=now + 30 * DAY_MS))
    second = runtime.service.status(row.owner, row.profile)["connections"][0]
    assert second["access_state"] == "renews_automatically"
    assert second["access_expires_at_ms"] == now + 30 * DAY_MS


@pytest.mark.parametrize("remaining_days", [29, 3, -2])
def test_sign_in_required_wins_over_every_expiry(tmp_path, remaining_days):
    connection, _ = completed_status(
        tmp_path, remaining_ms=int(remaining_days * DAY_MS), transport="authorization_required"
    )
    assert connection["access_state"] == "sign_in_required"


def test_live_connection_reports_access_state(tmp_path):
    connection, _ = completed_status(
        tmp_path, remaining_ms=20 * DAY_MS, transport="transport_ready", verified=True
    )
    assert connection["state"] == "ready_for_client"
    assert connection["access_state"] == "renews_automatically"


def test_sign_in_link_rows_do_not_report_access_state(tmp_path):
    runtime, row = enrolled_runtime(tmp_path)
    connection = runtime.service.status(row.owner, row.profile)["connections"][0]
    assert connection["sign_in_state"] == "required"
    assert "access_state" not in connection and "access_expires_at_ms" not in connection


def test_access_status_exposes_no_secrets(tmp_path):
    _, status = completed_status(tmp_path, remaining_ms=2 * DAY_MS, transport="transport_ready")
    text = json.dumps(status)
    for forbidden in ("synthetic", "PRIVATE KEY", "access_token", "verifier", str(tmp_path)):
        assert forbidden not in text


def test_status_route_carries_access_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from superlocalmemory.server.routes.connections import router

    runtime, row = enrolled_runtime(tmp_path)
    runtime.store.save(
        replace(row, completed=True, expires_at_ms=int(time.time() * 1000) + 3 * DAY_MS)
    )
    app = FastAPI()
    app.include_router(router)
    app.state.profile_runtime = SimpleNamespace(snapshot=SimpleNamespace(profile_id="profile"))
    app.state.daemon_descriptor = SimpleNamespace(port=9999)
    app.state.remote_connections = runtime.service
    with TestClient(app, base_url="http://127.0.0.1:9999", client=("127.0.0.1", 5000)) as client:
        body = client.get("/api/v3/connections/status")
    assert body.status_code == 200
    connection = body.json()["connections"][0]
    assert connection["access_state"] == "ending_soon"
    assert isinstance(connection["access_expires_at_ms"], int)
    assert "synthetic" not in body.text and "PRIVATE KEY" not in body.text
