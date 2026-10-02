# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`session_init`'s structured fields (``abstained``, ``abstention_reason``,
``answer_confidence``, ``calibration_status``, ``calibration_id``) already
pass through unchanged — see the JSON-shape assertions below. What was
missing: the ``context`` STRING is the part many hosts actually inject into
the conversation; a host that only reads ``context`` (not the sibling JSON
fields) saw the memory list with no indication the judge called it
insufficient. These tests pin a one-line prefix on ``context`` for the
judged-insufficient and judged-and-answered cases, and no change at all when
no judge is configured (calibration_status == "uncalibrated").
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from superlocalmemory.mcp._pool_adapter import (
    PoolFact,
    PoolRecallItem,
    PoolRecallResponse,
)


class _MockServer:
    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


def _get_session_init_tool():
    from superlocalmemory.mcp.tools_active import register_active_tools

    srv = _MockServer()
    get_engine = MagicMock()
    register_active_tools(srv, get_engine)
    return srv._tools["session_init"], get_engine


def _make_engine_mock(profile_id="default", feedback_count=10):
    engine = MagicMock()
    engine.profile_id = profile_id
    engine._adaptive_learner.get_feedback_count.return_value = feedback_count
    return engine


def _make_rules_mock(threshold=0.3):
    rules = MagicMock()
    rules.should_recall.return_value = True
    rules.get_recall_config.return_value = {
        "enabled": True, "relevance_threshold": threshold,
        "max_memories_injected": 10,
    }
    return rules


def _make_response(**overrides) -> PoolRecallResponse:
    base = dict(
        results=[PoolRecallItem(
            fact=PoolFact(fact_id="f-0", content="JWT with 1h expiry", memory_id="m-0"),
            score=0.9,
        )],
        calibration_status="uncalibrated",
        calibration_id=None,
        answer_confidence=None,
        abstained=False,
        abstention_reason=None,
    )
    base.update(overrides)
    return PoolRecallResponse(**base)


class TestSessionInitAnswerCheck:
    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent", create=True)
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_judged_insufficient_prefixes_context_with_warning(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        MockRulesEngine.return_value = _make_rules_mock()
        mock_pool_recall.return_value = _make_response(
            calibration_status="shadow", abstained=True,
            abstention_reason="judged_insufficient", answer_confidence=0.04,
        )
        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = _make_engine_mock()

        result = asyncio.run(session_init(project_path="/my/project"))

        assert result["abstained"] is True
        assert result["abstention_reason"] == "judged_insufficient"
        assert result["answer_confidence"] == 0.04
        assert "none of these memories answers the question" in result["context"]
        assert "0.04" in result["context"]

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent", create=True)
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_judged_and_answered_prefixes_context_with_confidence_note(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        MockRulesEngine.return_value = _make_rules_mock()
        mock_pool_recall.return_value = _make_response(
            calibration_status="shadow", abstained=False, answer_confidence=0.86,
        )
        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = _make_engine_mock()

        result = asyncio.run(session_init(project_path="/my/project"))

        assert "Answer check: likely answered" in result["context"]
        assert "0.86" in result["context"]
        # Original memory content must still be present — abstention is a
        # signal, never a suppression of the underlying results.
        assert "JWT with 1h expiry" in result["context"]

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent", create=True)
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_no_judge_configured_context_unchanged(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        MockRulesEngine.return_value = _make_rules_mock()
        mock_pool_recall.return_value = _make_response()  # uncalibrated
        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = _make_engine_mock()

        result = asyncio.run(session_init(project_path="/my/project"))

        assert "Answer check" not in result["context"]
        assert "JWT with 1h expiry" in result["context"]
