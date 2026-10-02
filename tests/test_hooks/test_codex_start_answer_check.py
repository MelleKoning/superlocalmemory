# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Codex's SessionStart hook (``_hook_codex_start``) auto-injects
``additionalContext`` into the conversation without the agent asking.

It calls the REAL ``session_init`` MCP tool over a subprocess JSON-RPC
pipe (``_codex_mcp_session_init``) — so the result it gets back already
carries the fixed, answer-check-prefixed ``context`` string and the raw
``abstained``/``abstention_reason`` fields (see
test_session_init_answer_check.py for that part). But ``_hook_codex_start``
never used ``session["context"]``: it builds its OWN ``context`` from a
separate, judge-blind fast-SQL path (``slm session-context <name>``, no
``--full``, confirmed by reading cli/commands.py:cmd_session_context), and
it summarizes the real session_init result into ``session_msg`` using only
``session_id`` / ``memory_count`` / ``retrieval_mode`` — silently dropping
``abstained`` and ``abstention_reason`` even though they were sitting right
there in the dict it already had. This is exactly the "summarised
session_init that drops fields" gap: the full MCP tool carries the signal,
but this one host-specific wrapper never reads it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from superlocalmemory.hooks import hook_handlers


def _mock_session_init_response(session_dict: dict):
    """Build the stdout lines _codex_mcp_session_init expects to read."""
    initialize_line = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n"
    tools_call_line = json.dumps({
        "jsonrpc": "2.0", "id": 2,
        "result": {"content": [{"type": "text", "text": json.dumps(session_dict)}]},
    }) + "\n"
    return [initialize_line, tools_call_line]


class TestCodexStartAnswerCheck:
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_judged_insufficient_surfaces_in_the_injected_context(
        self, mock_run, mock_popen, capsys,
    ):
        mock_run.return_value = MagicMock(stdout="some fast-path context", returncode=0)
        stdin = MagicMock()
        stdout = MagicMock()
        stdout.readline.side_effect = _mock_session_init_response({
            "session_id": "slm-20260101-aaaaaaaa",
            "memory_count": 2,
            "retrieval_mode": "hybrid_candidate_fusion",
            "context": "(the fixed context, unused by this host wrapper)",
            "abstained": True,
            "abstention_reason": "judged_insufficient",
            "answer_confidence": 0.04,
        })
        mock_popen.return_value.stdin = stdin
        mock_popen.return_value.stdout = stdout

        hook_handlers._hook_codex_start()

        payload = json.loads(capsys.readouterr().out)
        combined = (
            payload["systemMessage"]
            + payload["hookSpecificOutput"]["additionalContext"]
        )
        assert "answer check" in combined.lower()
        assert "none of these memories answers" in combined.lower()

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_judged_and_answered_unchanged_summary(self, mock_run, mock_popen, capsys):
        """A judge that found a confident answer adds no warning — the
        existing memory-count summary is already an accurate signal."""
        mock_run.return_value = MagicMock(stdout="some fast-path context", returncode=0)
        stdin = MagicMock()
        stdout = MagicMock()
        stdout.readline.side_effect = _mock_session_init_response({
            "session_id": "slm-20260101-aaaaaaaa",
            "memory_count": 1,
            "retrieval_mode": "hybrid_candidate_fusion",
            "context": "(context)",
            "abstained": False,
            "abstention_reason": None,
            "answer_confidence": 0.86,
        })
        mock_popen.return_value.stdin = stdin
        mock_popen.return_value.stdout = stdout

        hook_handlers._hook_codex_start()

        payload = json.loads(capsys.readouterr().out)
        combined = (
            payload["systemMessage"]
            + payload["hookSpecificOutput"]["additionalContext"]
        )
        assert "answer check" not in combined.lower()
        assert "SLM session_init OK" in combined

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_no_judge_configured_unchanged(self, mock_run, mock_popen, capsys):
        mock_run.return_value = MagicMock(stdout="some fast-path context", returncode=0)
        stdin = MagicMock()
        stdout = MagicMock()
        stdout.readline.side_effect = _mock_session_init_response({
            "session_id": "slm-20260101-aaaaaaaa",
            "memory_count": 2,
            "retrieval_mode": "hybrid_candidate_fusion",
            "context": "(context)",
            "abstained": False,
            "abstention_reason": None,
        })
        mock_popen.return_value.stdin = stdin
        mock_popen.return_value.stdout = stdout

        hook_handlers._hook_codex_start()

        payload = json.loads(capsys.readouterr().out)
        combined = (
            payload["systemMessage"]
            + payload["hookSpecificOutput"]["additionalContext"]
        )
        assert "answer check" not in combined.lower()
        assert "SLM session_init OK" in combined
