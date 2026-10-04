# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm recall` prints results, not timing.

A fresh install printed "SpreadingActivation.search completed via daemon
(139ms)" above every result list: an internal class name and a timing figure
the user did not ask for. Timing belongs in the debug log; stdout carries the
results (and, with ``--json``, nothing but the JSON document).
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from unittest.mock import patch


def _args(**kwargs) -> Namespace:
    defaults = dict(
        query="which database does staging use", limit=10, json=False,
        fast=False, include_global=None, include_shared=None, window="",
        as_of="", known_as_of="", valid_at="", include_unknown=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


_RESULT = {
    "results": [{"fact_id": "f1", "content": "Staging runs Postgres 16", "score": 0.54}],
    "retrieval_time_ms": 139.0,
    "no_confident_match": False,
    "answer_check_status": "off",
}


def _recall(args: Namespace) -> None:
    from superlocalmemory.cli.commands import cmd_recall

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", return_value=dict(_RESULT)),
    ):
        cmd_recall(args)


def test_text_output_is_only_the_results(capsys, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="superlocalmemory.cli.commands"):
        _recall(_args())

    out = capsys.readouterr().out
    assert out.splitlines() == ["  1. [0.54] Staging runs Postgres 16"]
    # The timing is still available to anyone who turns on debug logging.
    assert any("139ms" in r.getMessage() for r in caplog.records)
    assert all(r.levelno == logging.DEBUG for r in caplog.records if "139ms" in r.getMessage())


def test_json_output_is_one_json_document(capsys) -> None:
    _recall(_args(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["results"][0]["content"] == "Staging runs Postgres 16"
