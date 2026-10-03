# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm kinds`` is a thin client of the dashboard's routes.

``daemon_request`` is bound to a ``TestClient`` of the real routes, so these
tests exercise the CLI and the HTTP surface together: what the terminal says
is what the dashboard would say.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import kinds_cmd
from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound
from superlocalmemory.core.memory_kind_config import memory_kind_config_from
from tests.test_server.test_memory_kinds_routes import _Judge, _client, _fact


def _bind(monkeypatch, tc) -> None:
    def fake_daemon_request(method, path, body=None, *, preserve_conflict=False,
                            preserve_not_found=False, **_kw):
        response = tc.request(method, path, json=body)
        if response.status_code == 409 and preserve_conflict:
            raise DaemonConflict(response.json().get("detail", ""))
        if response.status_code == 404 and preserve_not_found:
            raise DaemonNotFound(404, "not_found", "daemon returned 404", path)
        if response.status_code >= 400:
            return None
        return response.json()

    monkeypatch.setattr(kinds_cmd, "daemon_request", fake_daemon_request)


def _args(**kw) -> Namespace:
    base = {"json": False, "kinds_command": None, "backfill_command": None}
    base.update(kw)
    return Namespace(**base)


def test_json_schema_stable(tmp_path, monkeypatch, capsys) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "Never push to main.")
    _bind(monkeypatch, tc)
    kinds_cmd.cmd_kinds(_args(json=True, kinds_command="status"))
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["success"] is True and envelope["command"] == "kinds status"
    data = envelope["data"]
    assert {"schema_ready", "enabled", "backend", "counts", "active_run",
            "recent_runs"} <= set(data)


def test_device_leaving_backfill_needs_yes(tmp_path, monkeypatch, capsys) -> None:
    cfg = memory_kind_config_from({"backend": "jev", "jev_consent": True})
    tc, app, db = _client(tmp_path, monkeypatch, judge=_Judge("jev"), cfg=cfg)
    _fact(db, "a memory")
    _bind(monkeypatch, tc)
    with pytest.raises(SystemExit) as exited:
        kinds_cmd.cmd_kinds(_args(kinds_command="backfill", backfill_command="start",
                                  mode="untyped", yes=False))
    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert "--yes" in out and "Jev" in out
    assert db.execute("SELECT COUNT(*) AS n FROM memory_kind_runs")[0]["n"] == 0
    kinds_cmd.cmd_kinds(_args(kinds_command="backfill", backfill_command="start",
                              mode="untyped", yes=True))
    assert "Started run" in capsys.readouterr().out
    assert db.execute("SELECT COUNT(*) AS n FROM memory_kind_runs")[0]["n"] == 1


def test_backfill_lifecycle_through_the_cli(tmp_path, monkeypatch, capsys) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    for t in ("Never push to main.", "We decided to use SQLite."):
        _fact(db, t)
    _bind(monkeypatch, tc)
    kinds_cmd.cmd_kinds(_args(json=True, kinds_command="backfill", backfill_command="start",
                              mode="untyped", yes=False))
    run_id = json.loads(capsys.readouterr().out)["data"]["run_id"]
    kinds_cmd.cmd_kinds(_args(kinds_command="backfill", backfill_command="pause",
                              run_id=run_id))
    assert "paused" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        kinds_cmd.cmd_kinds(_args(kinds_command="backfill", backfill_command="pause",
                                  run_id="no-such-run"))
    assert "No such" in capsys.readouterr().out


def test_settings_change_and_show(tmp_path, monkeypatch, capsys) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _bind(monkeypatch, tc)
    kinds_cmd.cmd_kinds(_args(kinds_command="settings", enable=False, disable=True,
                              backend="rules", jev_consent=None, standing_rules=None))
    out = capsys.readouterr().out
    assert "enabled: False" in out and "backend: rules" in out


def test_daemon_not_running(monkeypatch, capsys) -> None:
    monkeypatch.setattr(kinds_cmd, "daemon_request", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        kinds_cmd.cmd_kinds(_args(kinds_command="status"))
    assert "slm serve" in capsys.readouterr().out
