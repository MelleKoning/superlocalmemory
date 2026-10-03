# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L3-14: every ``--json`` path prints valid JSON, on success AND on error.

Covers remember/recall's refused-before-any-request paths, the dispatch-level
DaemonRefused handler, and ``slm db restore`` / ``prepare-downgrade``'s
refusal paths. Each test runs the command's error path with --json and
json.loads()s the captured stdout.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest


def test_remember_invalid_kind_json(capsys) -> None:
    from superlocalmemory.cli import commands

    with pytest.raises(SystemExit) as exited:
        commands.cmd_remember(Namespace(content="x", tags="", kind="gossip", replaces=None,
                                        json=True, sync_mode=False, scope=None,
                                        shared_with=None))
    assert exited.value.code == 2
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False
    assert out["error"]["code"] == "INVALID_KIND"


def test_remember_empty_replaces_json(capsys) -> None:
    from superlocalmemory.cli import commands

    with pytest.raises(SystemExit) as exited:
        commands.cmd_remember(Namespace(content="x", tags="", kind="", replaces="  ",
                                        json=True, sync_mode=False, scope=None,
                                        shared_with=None))
    assert exited.value.code == 2
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False
    assert "replaces" in out["error"]["message"]


def test_recall_invalid_kind_json(capsys) -> None:
    from superlocalmemory.cli import commands

    with pytest.raises(SystemExit) as exited:
        commands.cmd_recall(Namespace(query="x", limit=5, json=True, kind="gossip"))
    assert exited.value.code == 2
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False
    assert out["error"]["code"] == "INVALID_KIND"


def test_dispatch_daemon_refusal_json(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import commands
    from superlocalmemory.cli.daemon import DaemonRefused

    def _refuse(args):
        raise DaemonRefused(403, "/api/memory-kinds/status")

    monkeypatch.setattr(commands, "_cmd_kinds_dispatch", _refuse)
    with pytest.raises(SystemExit) as exited:
        commands.dispatch(Namespace(command="kinds", json=True))
    assert exited.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False
    assert out["error"]["code"] == "NOT_AUTHORIZED"


def test_db_restore_missing_point_id_json(tmp_path, capsys) -> None:
    from superlocalmemory.cli import upgrade_cmd

    rc = upgrade_cmd.cmd_db_restore(Namespace(point_id=None, json=True, data_root=tmp_path,
                                              cancel=False))
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert "error" in out


def test_db_restore_unknown_point_json(tmp_path, capsys) -> None:
    from superlocalmemory.cli import upgrade_cmd

    rc = upgrade_cmd.cmd_db_restore(Namespace(point_id="nope", json=True, data_root=tmp_path,
                                              cancel=False, yes=True))
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["restored"] is False


def test_prepare_downgrade_no_yes_non_tty_json(tmp_path, monkeypatch, capsys) -> None:
    from superlocalmemory.cli import upgrade_cmd

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = upgrade_cmd.cmd_db_prepare_downgrade(
        Namespace(json=True, data_root=tmp_path, cancel=False, yes=False))
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["confirmed"] is False
