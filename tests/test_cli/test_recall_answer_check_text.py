# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm recall` (plain-text output) must surface the answer-check judge verdict.

The daemon's ``/recall`` route already returns ``abstained``,
``abstention_reason``, ``answer_confidence`` and ``calibration_status`` in its
JSON envelope (``recall_response_metadata``), and ``cmd_recall``'s ``--json``
branch already passes that envelope through unchanged. The plain-text branch
printed the result list with no indication that the judge had run at all —
a judged-insufficient recall looked identical to a confident one. These tests
pin the three cases: judged-insufficient (warn), judged-and-answered (short
confidence note), and no judge configured (unchanged, no new line).
"""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch


def _args(**kwargs) -> Namespace:
    defaults = dict(
        query="what auth strategy did we pick",
        limit=10,
        json=False,
        fast=False,
        include_global=None,
        include_shared=None,
        window="",
        as_of="",
        known_as_of="",
        valid_at="",
        include_unknown=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def _daemon_result(**overrides) -> dict:
    base = {
        "results": [{"fact_id": "f1", "content": "JWT with 1h expiry", "score": 0.81}],
        "retrieval_time_ms": 42.0,
        "no_confident_match": False,
        "calibration_status": "uncalibrated",
        "calibration_id": None,
        "answer_confidence": None,
        "abstained": False,
        "abstention_reason": None,
    }
    base.update(overrides)
    return base


class TestRecallTextAnswerCheck:
    def test_judged_insufficient_prints_warning_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_recall

        result = _daemon_result(
            calibration_status="shadow",
            abstained=True,
            abstention_reason="judged_insufficient",
            answer_confidence=0.04,
        )
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", return_value=result),
        ):
            cmd_recall(_args())

        out = capsys.readouterr().out
        assert "Answer check" in out
        assert "none of these memories answers the question" in out
        assert "0.04" in out
        # The underlying (misleading-looking) result list still prints —
        # abstention is a signal, results are never removed.
        assert "JWT with 1h expiry" in out

    def test_judged_and_answered_prints_confidence_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_recall

        result = _daemon_result(
            calibration_status="shadow",
            abstained=False,
            abstention_reason=None,
            answer_confidence=0.86,
        )
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", return_value=result),
        ):
            cmd_recall(_args())

        out = capsys.readouterr().out
        assert "Answer check: likely answered" in out
        assert "0.86" in out

    def test_no_judge_configured_prints_no_answer_check_line(self, capsys) -> None:
        """uncalibrated (no judge) must be byte-for-byte unchanged output."""
        from superlocalmemory.cli.commands import cmd_recall

        result = _daemon_result()  # calibration_status="uncalibrated" default
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", return_value=result),
        ):
            cmd_recall(_args())

        out = capsys.readouterr().out
        assert "Answer check" not in out

    def test_json_mode_passes_fields_through_unchanged(self, capsys) -> None:
        """--json must still be a raw pass-through — no new text injected."""
        import json as _json

        from superlocalmemory.cli.commands import cmd_recall

        result = _daemon_result(
            calibration_status="shadow",
            abstained=True,
            abstention_reason="judged_insufficient",
            answer_confidence=0.04,
        )
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", return_value=result),
        ):
            cmd_recall(_args(json=True))

        payload = _json.loads(capsys.readouterr().out)
        data = payload["data"]
        assert data["abstained"] is True
        assert data["abstention_reason"] == "judged_insufficient"
        assert data["answer_confidence"] == 0.04
        assert data["calibration_status"] == "shadow"
