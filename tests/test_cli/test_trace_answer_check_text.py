# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm trace` (plain-text output, both daemon and direct-engine paths) must
surface the answer-check judge verdict the same way `slm recall` does.

See test_recall_answer_check_text.py for the shared rationale. This file
covers cmd_trace's TWO text-printing branches: the daemon-backed path (JSON
dict from ``/api/v3/recall/trace``) and the direct-engine fallback path (a
real-shaped ``RecallResponse`` object, used only when the daemon raised).
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch


def _args(**kwargs) -> Namespace:
    defaults = dict(query="auth strategy", limit=10, json=False)
    defaults.update(kwargs)
    return Namespace(**defaults)


def _daemon_result(**overrides) -> dict:
    base = {
        "query": "auth strategy",
        "query_type": "factual",
        "retrieval_time_ms": 12.0,
        "results": [{
            "fact_id": "f1", "content": "JWT with 1h expiry",
            "score": 0.81, "relevance_score": 0.81, "channel_scores": {},
        }],
        "no_confident_match": False,
        "calibration_status": "uncalibrated",
        "calibration_id": None,
        "answer_confidence": None,
        "abstained": False,
        "abstention_reason": None,
    }
    base.update(overrides)
    return base


def _fake_response(**overrides):
    fact = SimpleNamespace(content="JWT with 1h expiry")
    result = SimpleNamespace(
        fact=fact, relevance_score=0.81, ranking_score=None, channel_scores={},
    )
    base = dict(
        query_type="factual",
        retrieval_time_ms=12.0,
        results=[result],
        score_contract_version="2",
        calibration_status="uncalibrated",
        calibration_id=None,
        answer_confidence=None,
        abstained=False,
        abstention_reason=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestTraceDaemonTextAnswerCheck:
    def test_judged_insufficient_prints_warning_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_trace

        result = _daemon_result(
            calibration_status="shadow", abstained=True,
            abstention_reason="judged_insufficient", answer_confidence=0.04,
        )
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", return_value=result),
        ):
            cmd_trace(_args())

        out = capsys.readouterr().out
        assert "none of these memories answers the question" in out
        assert "0.04" in out

    def test_judged_and_answered_prints_confidence_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_trace

        result = _daemon_result(
            calibration_status="shadow", abstained=False, answer_confidence=0.86,
        )
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", return_value=result),
        ):
            cmd_trace(_args())

        out = capsys.readouterr().out
        assert "Answer check: likely answered" in out
        assert "0.86" in out

    def test_no_judge_configured_prints_no_answer_check_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_trace

        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", return_value=_daemon_result()),
        ):
            cmd_trace(_args())

        assert "Answer check" not in capsys.readouterr().out


class TestTraceDirectEngineFallbackTextAnswerCheck:
    """Daemon raises -> cmd_trace falls back to a direct engine.recall()."""

    def test_judged_insufficient_prints_warning_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_trace

        response = _fake_response(
            calibration_status="shadow", abstained=True,
            abstention_reason="judged_insufficient", answer_confidence=0.04,
        )
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=False),
            patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=False),
            patch("superlocalmemory.core.config.SLMConfig.load"),
            patch("superlocalmemory.core.engine.MemoryEngine") as mock_engine_cls,
        ):
            mock_engine_cls.return_value.recall.return_value = response
            cmd_trace(_args())

        out = capsys.readouterr().out
        assert "none of these memories answers the question" in out
        assert "0.04" in out

    def test_judged_and_answered_prints_confidence_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_trace

        response = _fake_response(
            calibration_status="shadow", abstained=False, answer_confidence=0.86,
        )
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=False),
            patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=False),
            patch("superlocalmemory.core.config.SLMConfig.load"),
            patch("superlocalmemory.core.engine.MemoryEngine") as mock_engine_cls,
        ):
            mock_engine_cls.return_value.recall.return_value = response
            cmd_trace(_args())

        out = capsys.readouterr().out
        assert "Answer check: likely answered" in out
        assert "0.86" in out

    def test_no_judge_configured_prints_no_answer_check_line(self, capsys) -> None:
        from superlocalmemory.cli.commands import cmd_trace

        response = _fake_response()  # calibration_status="uncalibrated" default
        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=False),
            patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=False),
            patch("superlocalmemory.core.config.SLMConfig.load"),
            patch("superlocalmemory.core.engine.MemoryEngine") as mock_engine_cls,
        ):
            mock_engine_cls.return_value.recall.return_value = response
            cmd_trace(_args())

        assert "Answer check" not in capsys.readouterr().out
