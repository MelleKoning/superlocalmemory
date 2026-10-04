# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm remember ... --replaces <id>``: a malformed id or a daemon refusal exits
2 without saving; a good one is sent to the daemon and what it replaced (or why
it could not) is printed - never left out."""

from __future__ import annotations

import io
import json
import urllib.error
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests._urlopen_fake import patch_urlopen

CONTENT = "The release train leaves on Thursday at 09:00 UTC."
OLD_ID = "3f2a9c0d11e84b7a"


def _args(replaces=None, *, as_json=False) -> Namespace:
    return Namespace(content=CONTENT, tags="", kind="", json=as_json, sync_mode=False,
                     scope=None, shared_with=None, replaces=replaces)


def _reply(replaced=None) -> dict:
    body = {"ok": True, "fact_ids": ["f-new"], "count": 1, "operation_id": "op-1",
            "materialization_state": "queryable"}
    if replaced is not None:
        body["replaced"] = replaced
    return body


def _run(args, reply=None, *, side_effect=None):
    from superlocalmemory.cli.commands import cmd_remember

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", return_value=reply,
              side_effect=side_effect) as request,
    ):
        cmd_remember(args)
    return request


@pytest.mark.parametrize("value", ["", "   ", "not an id", "-flag"])
def test_a_malformed_replaces_exits_2_before_contacting_slm(value, capsys) -> None:
    from superlocalmemory.cli import commands, daemon

    with (
        patch.object(daemon, "is_daemon_running", side_effect=AssertionError("contacted")),
        patch.object(daemon, "daemon_request", side_effect=AssertionError("contacted")),
        pytest.raises(SystemExit) as stopped,
    ):
        commands.cmd_remember(_args(value))
    assert stopped.value.code == 2
    assert "replaces" in capsys.readouterr().err


def test_replaces_is_sent_and_what_it_replaced_is_printed(capsys) -> None:
    request = _run(_args(f" {OLD_ID} "), _reply({"ok": True, "replaces": OLD_ID,
                                                 "fact_ids": [OLD_ID], "cases": [],
                                                 "undo": "x"}))
    body = request.call_args.args[2]
    assert body["replaces"] == OLD_ID
    assert request.call_args.kwargs["preserve_unprocessable"] is True
    out = capsys.readouterr().out
    assert "Replaced" in out and OLD_ID in out


def test_a_failed_mark_is_reported_on_stderr(capsys) -> None:
    _run(_args(OLD_ID), _reply({"ok": False, "replaces": OLD_ID,
                                "reason": "Nothing current to replace."}))
    captured = capsys.readouterr()
    assert "Nothing current to replace." in captured.err
    assert "NOT replaced" in captured.err


def test_json_output_carries_replaced(capsys) -> None:
    replaced = {"ok": False, "replaces": OLD_ID, "reason": "busy"}
    _run(_args(OLD_ID, as_json=True), _reply(replaced))
    assert json.loads(capsys.readouterr().out)["data"]["replaced"] == replaced


def test_a_daemon_refusal_exits_2_with_its_message(capsys) -> None:
    from superlocalmemory.cli.daemon import DaemonUnprocessable

    refusal = DaemonUnprocessable("REPLACES_NOT_FOUND", "No memory with id x in this profile.")
    with pytest.raises(SystemExit) as stopped:
        _run(_args(OLD_ID), side_effect=refusal)
    assert stopped.value.code == 2
    assert "No memory with id x in this profile." in capsys.readouterr().err


def test_without_replaces_the_request_is_unchanged() -> None:
    request = _run(_args(None), _reply())
    assert "replaces" not in request.call_args.args[2]
    assert "preserve_unprocessable" not in request.call_args.kwargs


def test_the_flag_reaches_the_command() -> None:
    from superlocalmemory.cli.main import main

    seen = {}
    with (
        patch("sys.argv", ["slm", "remember", CONTENT, "--replaces", OLD_ID]),
        patch("superlocalmemory.cli.setup_wizard.check_first_use"),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch("superlocalmemory.cli.commands.dispatch", side_effect=lambda a: seen.update(vars(a))),
    ):
        main()
    assert seen["replaces"] == OLD_ID


# --- the daemon client: a 422 is an answer, not an outage --------------------

def _http_422(body: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:1/remember", 422, "Unprocessable",
                                  {}, io.BytesIO(json.dumps(body).encode()))


def _post(error, **flags):
    from superlocalmemory.cli.daemon import daemon_request

    descriptor = SimpleNamespace(port=1, capability="cap", instance_id="inst")
    with patch_urlopen(side_effect=error):
        return daemon_request("POST", "/remember", {"content": "x"},
                              expected_descriptor=descriptor, verify_health=False, **flags)


def test_a_422_is_raised_with_its_code_when_asked() -> None:
    from superlocalmemory.cli.daemon import DaemonUnprocessable

    with pytest.raises(DaemonUnprocessable) as refused:
        _post(_http_422({"detail": {"code": "REPLACES_NOT_ALLOWED", "message": "theirs"}}),
              preserve_unprocessable=True)
    assert (refused.value.code, refused.value.message) == ("REPLACES_NOT_ALLOWED", "theirs")
    with pytest.raises(DaemonUnprocessable) as plain:
        _post(_http_422({"detail": "Unknown memory kind."}), preserve_unprocessable=True)
    assert plain.value.message == "Unknown memory kind."


def test_a_422_keeps_its_old_meaning_when_not_asked() -> None:
    assert _post(_http_422({"detail": "x"})) is None


def test_the_undo_hint_prints_each_exact_rollback_command(capsys) -> None:
    """Audit 4.1.20 L2: the hint named no case_id or version, so there was
    nothing a person could type. Each case now gets its literal command."""
    _run(_args(OLD_ID), _reply({
        "ok": True, "replaces": OLD_ID, "fact_ids": [OLD_ID, "f-2"],
        "cases": [{"case_id": "case-aaa", "version": 1},
                  {"case_id": "case-bbb", "version": 3}],
        "undo": "To undo, call review_correction with each case_id ...",
    }))
    out = capsys.readouterr().out
    assert "slm review-correction case-aaa rollback 1" in out
    assert "slm review-correction case-bbb rollback 3" in out
    assert "call review_correction with each case_id" not in out
