# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""An unknown memory id is NOT_FOUND, never DAEMON_UNAVAILABLE (audit M4)."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from superlocalmemory.cli.daemon import DaemonNotFound


def _not_found(*_a, **kwargs):
    # A real daemon only raises when the caller asked to preserve the 404;
    # otherwise daemon_request collapses it to None.
    if kwargs.get("preserve_not_found"):
        raise DaemonNotFound(404, "not_found", "Memory not found", "/x")
    return None


def _run(cmd, args):
    with patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True), \
         patch("superlocalmemory.cli.daemon.daemon_request", side_effect=_not_found), \
         patch("superlocalmemory.cli.commands._daemon_unavailable",
               side_effect=AssertionError("a live daemon's 404 is not an outage")), \
         pytest.raises(SystemExit) as exc:
        cmd(args)
    return exc.value.code


@pytest.mark.parametrize("use_json", [False, True])
@pytest.mark.parametrize("confirmed", [False, True])
def test_delete_unknown_id(capsys, use_json, confirmed) -> None:
    from superlocalmemory.cli.commands import cmd_delete

    code = _run(cmd_delete, Namespace(fact_id="nope-123", yes=confirmed, json=use_json))
    out, err = capsys.readouterr()
    assert code == 1
    if use_json:
        payload = json.loads(out)
        assert payload["success"] is False
        assert payload["error"]["code"] == "NOT_FOUND"
        assert payload["error"]["message"] == "No memory with id nope-123"
    else:
        assert "No memory with id nope-123" in err
        assert "DAEMON_UNAVAILABLE" not in out + err


@pytest.mark.parametrize("use_json", [False, True])
def test_update_unknown_id(capsys, use_json) -> None:
    from superlocalmemory.cli.commands import cmd_update

    code = _run(cmd_update, Namespace(fact_id="nope-123", content="x", json=use_json))
    out, err = capsys.readouterr()
    assert code == 1
    if use_json:
        assert json.loads(out)["error"]["code"] == "NOT_FOUND"
    else:
        assert "No memory with id nope-123" in err
