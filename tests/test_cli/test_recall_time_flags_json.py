# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm recall --json` with a time filter prints JSON, on success AND on error.

The time filters are --window, --as-of, --known-as-of and --valid-at. Before
4.1.20 a bad value for any of the last three was reported as plain text on
stderr even under --json, so a script parsing stdout got nothing; a bad
--window was dropped without a word and the recall ran unfiltered.

The daemon is never contacted here: it is patched, and the invalid cases
must be refused before it is even probed.
"""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest


def _args(**kwargs) -> Namespace:
    base = dict(query="what changed", limit=5, json=True, kind="", fast=False,
                include_global=None, include_shared=None, window="", as_of="",
                known_as_of="", valid_at="", include_unknown=False,
                project="", saved_by="", about="")
    base.update(kwargs)
    return Namespace(**base)


def _no_daemon():
    def _boom(*_a, **_k):
        raise AssertionError("an invalid time filter must not reach the daemon")
    return (
        patch("superlocalmemory.cli.daemon.is_daemon_running", side_effect=_boom),
        patch("superlocalmemory.cli.daemon.ensure_daemon", side_effect=_boom),
        patch("superlocalmemory.cli.daemon.daemon_request", side_effect=_boom),
    )


BAD = [
    ("window", "--window", "fortnight-ish"),
    ("window", "--window", "2026-07-01..not-a-date"),
    ("as_of", "--as-of", "yesterday-ish"),
    ("known_as_of", "--known-as-of", "13/45/2026"),
    ("valid_at", "--valid-at", "soon"),
]


@pytest.mark.parametrize("attr, flag, value", BAD, ids=[b[1] + ":" + b[2] for b in BAD])
def test_bad_time_filter_is_json_under_json(attr, flag, value, capsys) -> None:
    from superlocalmemory.cli.commands import cmd_recall

    p1, p2, p3 = _no_daemon()
    with p1, p2, p3, pytest.raises(SystemExit) as exited:
        cmd_recall(_args(**{attr: value}))
    assert exited.value.code == 1
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["success"] is False and out["command"] == "recall"
    assert out["error"]["code"] == "INVALID_TIME_FILTER"
    assert out["error"]["flag"] == flag
    assert value in out["error"]["message"]
    assert captured.err == ""


@pytest.mark.parametrize("attr, flag, value", BAD, ids=[b[1] + ":" + b[2] for b in BAD])
def test_bad_time_filter_without_json_stays_text_on_stderr(attr, flag, value, capsys) -> None:
    from superlocalmemory.cli.commands import cmd_recall

    p1, p2, p3 = _no_daemon()
    with p1, p2, p3, pytest.raises(SystemExit) as exited:
        cmd_recall(_args(json=False, **{attr: value}))
    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Error: invalid {flag} value: {value!r}" in captured.err


GOOD = [
    {"window": "7d"},
    {"window": "2026-07-01..2026-07-31"},
    {"as_of": "2026-01-01T00:00:00Z"},
    {"known_as_of": "2026-01-01T00:00:00+00:00", "valid_at": "2026-01-01", "include_unknown": True},
]


@pytest.mark.parametrize("flags", GOOD, ids=[",".join(g) for g in GOOD])
def test_good_time_filter_returns_a_json_envelope(flags, capsys) -> None:
    from superlocalmemory.cli.commands import cmd_recall

    seen: list[str] = []

    def _request(method, path, *a, **k):
        seen.append(path)
        return {"results": [{"content": "x", "score": 0.5}], "retrieval_time_ms": 3}

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", side_effect=_request),
    ):
        cmd_recall(_args(**flags))
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is True
    assert out["data"]["results"][0]["content"] == "x"
    assert len(seen) == 1
    for key in flags:
        if key != "include_unknown":
            assert f"&{key}=" in seen[0], seen[0]
