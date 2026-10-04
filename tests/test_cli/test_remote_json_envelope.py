# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm remote check --json`` and ``keys list --json`` use the standard CLI
envelope (audit 4.1.20 L4): every other ``--json`` command answers
``{"success", "command", "version", "data"}``; these printed a bare object an
agent could not parse the same way."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from superlocalmemory.cli import remote_commands


def test_check_json_is_the_standard_envelope(capsys) -> None:
    results = [{"status": "PASS", "check": "listener", "detail": "ok"},
               {"status": "WARN", "check": "keys", "detail": "no active keys"}]
    with patch.object(remote_commands, "run_checks", return_value=results):
        remote_commands.cmd_remote(Namespace(remote_command="check", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["command"] == "remote check"
    assert payload["version"]
    assert payload["data"]["results"] == results
    assert payload["data"]["summary"] == {"pass": 1, "warn": 1, "fail": 0, "info": 0}


def test_check_json_still_exits_1_on_a_failure(capsys) -> None:
    results = [{"status": "FAIL", "check": "key store", "detail": "unreadable"}]
    with patch.object(remote_commands, "run_checks", return_value=results), \
         pytest.raises(SystemExit) as exc:
        remote_commands.cmd_remote(Namespace(remote_command="check", json=True))
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["data"]["summary"]["fail"] == 1


def test_keys_list_json_is_the_standard_envelope(capsys) -> None:
    row = {"name": "hermes", "key_id": "k1", "scope": "read",
           "created_at": "2026-10-04", "revoked_at": None}
    store = SimpleNamespace(list=lambda: [SimpleNamespace(public=lambda: row)],
                            bind_unbound=lambda _profile: ())
    with patch("superlocalmemory.server.remote_keys.default_store", return_value=store):
        remote_commands.cmd_remote(
            Namespace(remote_command="keys", keys_command="list", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["command"] == "remote keys list"
    assert payload["data"]["keys"] == [row]
