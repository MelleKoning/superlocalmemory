# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The daemon health contract must expose verifiable process identity."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request

from superlocalmemory.infra.daemon_identity import (
    DAEMON_PROTOCOL,
    DAEMON_SERVICE,
    capability_fingerprint,
    namespace_id_for,
)


def test_health_preserves_compatibility_and_adds_owned_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43127")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "health-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "health-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)

    app = unified_daemon.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/health")
    payload = asyncio.run(route.endpoint())

    assert payload["status"] == "ok"
    assert payload["engine"] in {"initialized", "unavailable"}
    assert "embedding_warm" in payload
    assert "recall_health" in payload
    assert payload["service"] == DAEMON_SERVICE
    assert payload["daemon_protocol"] == DAEMON_PROTOCOL
    assert payload["namespace_id"] == namespace_id_for(tmp_path)
    assert payload["instance_id"] == "health-instance"
    assert payload["capability_fingerprint"] == capability_fingerprint(
        "health-capability"
    )
    assert payload["port"] == 43127


def _request(*, capability: str = "", instance_id: str = "") -> Request:
    headers = []
    if capability:
        headers.append((b"x-slm-daemon-capability", capability.encode()))
    if instance_id:
        headers.append((b"x-slm-target-instance", instance_id.encode()))
    return Request({"type": "http", "method": "POST", "path": "/stop", "headers": headers})


def test_stop_rejects_missing_or_wrong_process_capability(
    tmp_path: Path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43131")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "stop-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "stop-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/stop")

    with pytest.raises(HTTPException) as missing:
        asyncio.run(route.endpoint(_request()))
    assert missing.value.status_code == 403

    with pytest.raises(HTTPException) as wrong_instance:
        asyncio.run(route.endpoint(_request(
            capability="stop-capability", instance_id="replacement",
        )))
    assert wrong_instance.value.status_code == 409


def test_stop_accepts_only_the_owned_process_instance(
    tmp_path: Path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43132")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "stop-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "stop-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/stop")

    with (
        patch.object(unified_daemon._observe_buffer, "flush_sync"),
        patch("os.kill") as kill,
        patch("signal.raise_signal") as raise_signal,
    ):
        payload = asyncio.run(route.endpoint(_request(
            capability="stop-capability", instance_id="stop-instance",
        )))

    assert payload == {"status": "stopping"}
    # In-process delivery, never os.kill(own pid): on Windows that is
    # TerminateProcess, which skips the shutdown that removes daemon.json.
    raise_signal.assert_called_once_with(signal.SIGTERM)
    kill.assert_not_called()


def test_a_shutdown_request_reaches_this_processs_handler_without_os_kill(
    monkeypatch,
) -> None:
    """The server's SIGTERM handler runs (graceful shutdown); ``os.kill`` on
    its own PID, a hard kill on Windows, is never used."""
    from superlocalmemory.server import unified_daemon

    hard_killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: hard_killed.append((pid, sig)))
    handled: list[int] = []
    previous = signal.signal(signal.SIGTERM, lambda sig, _frame: handled.append(sig))
    try:
        unified_daemon._request_graceful_shutdown()
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert handled == [signal.SIGTERM]
    assert hard_killed == []


def test_legacy_redirect_targets_the_actual_runtime_port() -> None:
    import inspect

    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.lifespan)
    # 4.1.22: through the default-root ownership rule, still aimed at
    # the port this daemon actually serves, and (4.1.22 polish) reporting its
    # real outcome back through a status dict for /status to read truthfully.
    assert "application.state.daemon_descriptor.port," in source
    assert "_LEGACY_PORT," in source
    assert "status=application.state.legacy_port_status," in source
    assert "_maybe_start_legacy_redirect(" in source


def test_status_reports_the_legacy_port_only_when_actually_bound(
    tmp_path: Path, monkeypatch,
) -> None:
    """The status route must report the truth, not the policy (4.1.22 polish):
    a daemon whose legacy-port bind resolved to ``bound: True`` reports the
    real port number and the real reason."""
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43135")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    app.state.legacy_port_status = {
        "bound": True, "port": 8767, "reason": "default data root",
    }
    route = next(route for route in app.routes if getattr(route, "path", None) == "/status")

    payload = asyncio.run(route.endpoint())

    assert payload["legacy_port"] == 8767
    assert payload["legacy_port_reason"] == "default data root"


def test_status_reports_null_legacy_port_when_not_bound(
    tmp_path: Path, monkeypatch,
) -> None:
    """A daemon that never took 8767 (a second data root, or the socket was
    already held by another process) must report ``null``, never the
    constant 8767 -- that was the 4.1.22 bug (it always reported 8767)."""
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43136")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    app.state.legacy_port_status = {
        "bound": False,
        "port": None,
        "reason": "this daemon serves /somewhere/else, not the default data root",
    }
    route = next(route for route in app.routes if getattr(route, "path", None) == "/status")

    payload = asyncio.run(route.endpoint())

    assert payload["legacy_port"] is None
    assert "not the default data root" in payload["legacy_port_reason"]


def test_status_legacy_port_defaults_to_null_before_startup_runs(
    tmp_path: Path, monkeypatch,
) -> None:
    """Calling the route function directly (as these unit tests do) bypasses
    ``lifespan`` entirely, so ``legacy_port_status`` is never set -- must not
    crash, and must still refuse to claim the port is bound."""
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43137")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/status")

    payload = asyncio.run(route.endpoint())

    assert payload["legacy_port"] is None
    assert payload["legacy_port_reason"] == "unknown"


def test_status_reports_the_actual_runtime_port(tmp_path: Path, monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43134")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "status-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "status-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    route = next(
        route for route in app.routes if getattr(route, "path", None) == "/status"
    )

    payload = asyncio.run(route.endpoint())
    assert payload["port"] == 43134
