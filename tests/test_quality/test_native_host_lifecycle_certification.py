"""an earlier stage native-host certification for the *shipped* Codex and Claude assets.

This is deliberately an isolated contract test, not a test of Varun's live
host configuration.  It exercises the exact stdio environment shape used by
the package, with generated facts and generated SQLite files only.  A passing
result proves the portable MCP/lifecycle contract; it does not claim that a
particular desktop build executed its vendor-owned hook runtime.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from superlocalmemory.mcp._pool_adapter import PoolFact, PoolRecallItem, PoolRecallResponse

REPO = Path(__file__).resolve().parents[2]
CODEX_CONFIG = REPO / "codex-plugin" / ".codex" / "config.toml"
CODEX_HOOKS = REPO / "codex-plugin" / "hooks" / "hooks.json"
CLAUDE_MCP = REPO / "plugin" / ".mcp.json"
CLAUDE_HOOKS = REPO / "plugin" / "hooks" / "hooks.json"


class _ToolServer:
    """Small FastMCP-shaped recorder without starting a real host process."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *args: Any, **kwargs: Any):
        del args, kwargs

        def decorate(fn: Any) -> Any:
            self.tools[fn.__name__] = fn
            return fn

        return decorate


class _Engine:
    """Read-only session-init engine fixture; it never opens the real store."""

    profile_id = "native-certification"
    mode = "B"

    def __init__(self) -> None:
        self.db = SimpleNamespace(get_pinned=lambda _profile_id: [])
        self._adaptive_learner = SimpleNamespace(get_feedback_count=lambda _profile_id: 0)


def _recall_response() -> PoolRecallResponse:
    return PoolRecallResponse(
        results=[
            PoolRecallItem(
                fact=PoolFact(
                    fact_id="cert-fact-current",
                    memory_id="cert-memory-current",
                    content="Current approved native host certification decision.",
                    created_at="2026-08-16T00:00:00Z",
                ),
                score=0.95,
            ),
            PoolRecallItem(
                fact=PoolFact(
                    fact_id="cert-fact-context",
                    memory_id="cert-memory-context",
                    content="Supporting context for the native host certification.",
                    created_at="2026-08-16T00:00:00Z",
                ),
                score=0.90,
            ),
        ]
    )


def _rules() -> SimpleNamespace:
    return SimpleNamespace(
        should_recall=lambda _event: True,
        get_recall_config=lambda: {"relevance_threshold": 0.3},
    )


def _create_truth_stores(root: Path) -> None:
    """Create only the minimal schemas read by BrainTruth v1."""
    with sqlite3.connect(root / "memory.db") as conn:
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                lifecycle TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE correction_cases (
                case_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO atomic_facts VALUES
                ('cert-fact-current', 'native-certification', 'active', '2026-08-16T00:00:00Z');
            INSERT INTO correction_cases VALUES
                ('cert-case-1', 'native-certification', 'proposed');
            """
        )
    with sqlite3.connect(root / "learning.db") as conn:
        conn.executescript(
            """
            CREATE TABLE learning_signals (profile_id TEXT NOT NULL, signal_type TEXT NOT NULL);
            CREATE TABLE agent_experiences (
                profile_id TEXT NOT NULL,
                experience_id TEXT NOT NULL,
                verification_authority TEXT NOT NULL
            );
            CREATE TABLE cognitive_turn_receipts (
                profile_id TEXT NOT NULL,
                receipt_id TEXT NOT NULL,
                state TEXT NOT NULL
            );
            CREATE TABLE external_evidence_receipts (
                profile_id TEXT NOT NULL,
                run_ref TEXT NOT NULL,
                run_state TEXT NOT NULL,
                demonstration INTEGER NOT NULL,
                eligible_for_learning INTEGER NOT NULL
            );
            INSERT INTO learning_signals VALUES ('native-certification', 'user_positive');
            """
        )


def _mcp_session_fact_ids(host_id: str, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """Run the exact stdio identity boundary with deterministic recall data."""
    from superlocalmemory.mcp.agent_context import get_current_agent_id
    from superlocalmemory.mcp.tools_active import register_active_tools

    monkeypatch.setenv("SLM_AGENT_ID", host_id)
    assert get_current_agent_id() == host_id
    server = _ToolServer()
    engine = _Engine()
    register_active_tools(server, lambda: engine)

    # Native lifecycle must keep the session tools on the same MCP surface.
    assert {"session_init", "close_session"} <= server.tools.keys()
    with (
        patch("superlocalmemory.hooks.rules_engine.RulesEngine", return_value=_rules()),
        patch("superlocalmemory.mcp._pool_adapter.pool_recall", return_value=_recall_response()),
        patch("superlocalmemory.mcp.tools_active._canonical_feedback_count", return_value=0),
        patch("superlocalmemory.mcp.tools_active._emit_event"),
    ):
        result = asyncio.run(
            server.tools["session_init"](
                project_path="/generated/native-host-certification",
                max_results=5,
            )
        )

    assert result["success"] is True
    assert result["degraded_mode"] is False
    assert result["retrieval_mode"] == "hybrid_candidate_fusion"
    assert result["session_id"].startswith("slm-")
    return tuple(item["fact_id"] for item in result["memories"])


def _mcp_brain_truth(host_id: str, root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Call the actual MCP BrainTruth adapter through its generated stores."""
    from superlocalmemory.mcp.agent_context import get_current_agent_id
    from superlocalmemory.mcp.tools_brain import register_brain_tools

    monkeypatch.setenv("SLM_AGENT_ID", host_id)
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    assert get_current_agent_id() == host_id
    server = _ToolServer()
    register_brain_tools(server, _Engine)
    result = asyncio.run(server.tools["get_brain_evidence_status"]())
    assert result["success"] is True
    assert result["brain_truth"]["contract"] == "superlocalmemory.brain-truth/v1"
    assert result["brain_truth"]["control_plane"] == "observation_only"
    return result["brain_truth"]


def _stable_truth(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Remove the expected per-call observation timestamp before equality."""
    stable = copy.deepcopy(snapshot)
    stable.pop("generated_at", None)
    return stable


def _hook_commands(path: Path, event: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        hook.get("command", "")
        for entry in payload["hooks"].get(event, [])
        for hook in entry.get("hooks", [])
        if hook.get("type") == "command"
    ]


def test_shipped_codex_and_claude_assets_declare_the_same_native_lifecycle() -> None:
    """Certify package assets, never the user's host-owned configuration."""
    import tomllib

    codex = tomllib.loads(CODEX_CONFIG.read_text(encoding="utf-8"))
    codex_server = codex["mcp_servers"]["superlocalmemory"]
    claude = json.loads(CLAUDE_MCP.read_text(encoding="utf-8"))
    claude_server = claude["mcpServers"]["superlocalmemory"]

    assert codex_server["args"] == ["mcp"]
    assert "slm" in str(codex_server["command"])
    assert claude_server["args"] == []
    assert "slm-launch" in str(claude_server["command"])
    # 4.1.3: neither host pins a profile. Forcing one from a plugin removed
    # tools the user had enabled; unset resolves to the raw server, so saying
    # nothing exposes more than "code" ever did.
    assert "SLM_MCP_PROFILE" not in codex_server["env"]
    assert "SLM_MCP_PROFILE" not in claude_server["env"]

    for hooks in (CODEX_HOOKS, CLAUDE_HOOKS):
        starts = _hook_commands(hooks, "SessionStart")
        stops = _hook_commands(hooks, "Stop")
        # Claude plugin hooks run slm through scripts/slm-run (4.1.20, #141).
        starts = [c.replace('/scripts/slm-run" hook', ' slm hook') for c in starts]
        stops = [c.replace('/scripts/slm-run" hook', ' slm hook') for c in stops]
        assert any("slm hook start" in command for command in starts), hooks
        assert any("slm hook stop" in command for command in stops), hooks


def test_native_stdio_hosts_return_identical_fact_ids_for_the_same_session_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q-08: host identity cannot change deterministic session recall IDs."""
    codex_ids = _mcp_session_fact_ids("codex", monkeypatch)
    claude_ids = _mcp_session_fact_ids("claude_code", monkeypatch)

    assert codex_ids == ("cert-fact-current", "cert-fact-context")
    assert claude_ids == codex_ids


def test_native_stdio_hosts_expose_identical_observation_only_brain_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BrainTruth v1 is host-neutral and never claims it trained retrieval."""
    _create_truth_stores(tmp_path)
    codex_truth = _mcp_brain_truth("codex", tmp_path, monkeypatch)
    claude_truth = _mcp_brain_truth("claude_code", tmp_path, monkeypatch)

    assert _stable_truth(codex_truth) == _stable_truth(claude_truth)
    assert codex_truth["memory_activity"]["facts_total"] == 1
    assert codex_truth["feedback"]["signals_total"] == 1
    assert codex_truth["correction_quality"]["cases_by_status"] == {"proposed": 1}


def test_codex_code_profile_exposes_lifecycle_and_brain_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the real product registration/filter path, not a copied tool list."""
    # The repository pins mcp==2.0.0.  Some contributor shells intentionally
    # carry mcp 1.x for unrelated tools, where the MCP 2 server module simply
    # does not exist.  That is an environment limitation, not a reason to
    # substitute a copied tool inventory; package/CI environments must run it.
    pytest.importorskip("mcp.server.mcpserver", reason="requires packaged mcp==2.0.0")
    monkeypatch.setenv("SLM_MCP_EMBEDDED", "1")
    monkeypatch.setenv("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
    monkeypatch.setenv("SLM_MCP_PROFILE", "code")
    module_name = "superlocalmemory.mcp.server"
    previous = sys.modules.pop(module_name, None)
    try:
        server_module = importlib.import_module(module_name)
        names = {tool.name for tool in server_module.server._tool_manager.list_tools()}
        assert {
            "session_init",
            "close_session",
            "get_brain_evidence_status",
        } <= names
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
