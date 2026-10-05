# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | Workstream C — Transport Flexibility

"""TDD tests for selectable transport in connect_ide() — Workstream C.

Protocol: run RED against unmodified code; implement; run GREEN.

Scenarios covered:
  C-1  Regression guard: default (no --transport) still writes stdio block
  C-2  --transport http writes native MCP HTTP block
  C-3  --transport http-mcp-remote writes the mcp-remote bridge block
  C-4  http + profile appends ?profile=<name> to URL
  C-5  http + custom daemon_port uses the custom port
  C-6  http with daemon unreachable: warns, still writes config (no crash)
  C-7  Invalid transport returns an error result
  C-8  http-mcp-remote block structure is correct (command, args)
  C-9  YAML-format IDE (continue) with http transport: falls back gracefully
  C-10 connect_many() propagates transport param
  C-11 CLI argparse: --transport and --port flags are declared on connect_p
  C-12 connect_ide returns transport key in result
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Source is resolved via PYTHONPATH (the test harness sets it), NOT sys.path injection:
# injecting the worktree src defeats RED isolation (tests would pass on 3.8.3 too).
# val-C blocker fixed — do NOT re-add sys.path.insert here.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from superlocalmemory.hooks.portable_kit import (  # noqa: E402
    IDE_MATRIX,
    connect_ide,
    connect_many,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# C-1 Regression guard: default (no transport arg) still emits stdio block
# ---------------------------------------------------------------------------


def test_default_transport_is_stdio(fake_home: Path) -> None:
    """connect_ide without --transport must produce the stdio block — zero regression."""
    result = connect_ide("cursor", home=fake_home)

    assert result["error"] is None
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]
    assert block["type"] == "stdio", f"Default transport changed: got {block!r}"
    assert block["command"] == "slm"
    assert block["args"] == ["mcp"]


# ---------------------------------------------------------------------------
# C-2 --transport http writes the native MCP HTTP block
# ---------------------------------------------------------------------------


def test_transport_http_writes_native_block(fake_home: Path) -> None:
    """transport='http' → {type: http, url: http://127.0.0.1:8765/mcp/}."""
    result = connect_ide("cursor", home=fake_home, transport="http")

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]

    assert block["type"] == "http", f"Expected http block, got: {block!r}"
    assert block["url"] == "http://127.0.0.1:8765/mcp/"
    # Must NOT contain stdio fields
    assert "command" not in block
    assert "args" not in block


# ---------------------------------------------------------------------------
# C-3 --transport http-mcp-remote writes the mcp-remote bridge block
# ---------------------------------------------------------------------------


def test_transport_http_mcp_remote_writes_bridge_block(fake_home: Path) -> None:
    """transport='http-mcp-remote' → stdio block with mcp-remote command."""
    result = connect_ide("gemini-cli", home=fake_home, transport="http-mcp-remote")

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    desc = IDE_MATRIX["gemini-cli"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]

    assert block["type"] == "stdio"
    assert block["command"] == "mcp-remote"
    assert isinstance(block["args"], list)
    assert len(block["args"]) >= 1
    assert "8765" in block["args"][0]
    assert "/mcp/" in block["args"][0]


# ---------------------------------------------------------------------------
# C-4 http + profile appends ?profile=<name> to URL
# ---------------------------------------------------------------------------


def test_transport_http_with_profile_appends_query_param(fake_home: Path) -> None:
    """transport='http' + profile='power' → URL contains ?profile=power."""
    result = connect_ide("cursor", home=fake_home, transport="http", profile="power")

    assert result["error"] is None
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]

    assert block["type"] == "http"
    assert "?profile=power" in block["url"], f"Profile missing from URL: {block['url']}"


# ---------------------------------------------------------------------------
# C-5 http + custom daemon_port uses the custom port
# ---------------------------------------------------------------------------


def test_transport_http_with_custom_port(fake_home: Path) -> None:
    """transport='http' + daemon_port=9000 → URL uses port 9000."""
    result = connect_ide("cursor", home=fake_home, transport="http", daemon_port=9000)

    assert result["error"] is None
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]

    assert block["type"] == "http"
    assert "9000" in block["url"], f"Custom port missing from URL: {block['url']}"
    assert "127.0.0.1:9000" in block["url"]


# ---------------------------------------------------------------------------
# C-6 http with daemon unreachable: warns but does NOT crash; config is written
# ---------------------------------------------------------------------------


def test_transport_http_daemon_unreachable_no_crash(
    fake_home: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """When daemon is unreachable for http transport, connect_ide() still writes
    the config and returns no error — only a warning is emitted."""
    # Patch _check_daemon_health to simulate daemon being down
    with patch(
        "superlocalmemory.hooks.portable_kit._check_daemon_health",
        return_value=False,
    ):
        result = connect_ide("cursor", home=fake_home, transport="http")

    # Config MUST still be written (no exception, no error)
    assert result["error"] is None, f"Unexpected error on unreachable daemon: {result['error']}"
    assert result["mcp_config"] in ("wrote", "merged", "unchanged")

    # The config file must exist with the http block
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    assert config_path.exists()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]
    assert block["type"] == "http"

    # A warning must have been printed to stderr
    captured = capsys.readouterr()
    assert captured.err, "Expected a warning on stderr when daemon is unreachable"


# ---------------------------------------------------------------------------
# C-7 Invalid transport returns an error result
# ---------------------------------------------------------------------------


def test_invalid_transport_returns_error(fake_home: Path) -> None:
    """transport='banana' is not a valid transport; expect an error result."""
    result = connect_ide("cursor", home=fake_home, transport="banana")

    assert result["mcp_config"] == "error"
    assert result["error"] is not None
    assert "transport" in result["error"].lower()


# ---------------------------------------------------------------------------
# C-8 http-mcp-remote: URL in args contains the correct daemon endpoint
# ---------------------------------------------------------------------------


def test_transport_mcp_remote_url_in_args(fake_home: Path) -> None:
    """mcp-remote args[0] should be exactly http://127.0.0.1:8765/mcp/."""
    result = connect_ide("cursor", home=fake_home, transport="http-mcp-remote")

    assert result["error"] is None
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]

    assert block["args"][0] == "http://127.0.0.1:8765/mcp/"


# ---------------------------------------------------------------------------
# C-9 YAML-format IDE (continue.dev) with http transport: falls back gracefully
# ---------------------------------------------------------------------------


def test_transport_http_yaml_ide_falls_back_gracefully(fake_home: Path) -> None:
    """YAML-format IDEs (continue.dev) don't support the http block structure.

    Expected: connect_ide does not crash and produces a valid result.
    The block should be the stdio default (YAML IDEs don't use the JSON http block).
    """
    result = connect_ide("continue", home=fake_home, transport="http")

    # Must not error out — fall back or warn but stay non-fatal
    assert result["mcp_config"] in ("wrote", "merged", "unchanged", "would_write"), (
        f"Unexpected mcp_config: {result['mcp_config']}"
    )
    assert result["ide"] == "continue"


# ---------------------------------------------------------------------------
# C-10 connect_many() propagates transport param
# ---------------------------------------------------------------------------


def test_connect_many_propagates_transport(fake_home: Path) -> None:
    """connect_many with transport='http' should propagate to each IDE call."""
    results = connect_many(
        ["cursor", "antigravity"],
        home=fake_home,
        transport="http",
    )

    assert len(results) == 2
    for r in results:
        assert r["error"] is None, f"Error for {r['ide']}: {r['error']}"
        desc = IDE_MATRIX[r["ide"]]
        config_path = fake_home / desc.mcp_path_global
        data = json.loads(config_path.read_text(encoding="utf-8"))
        block = data[desc.server_key]["superlocalmemory"]
        assert block["type"] == "http", f"IDE {r['ide']} did not get http block"


# ---------------------------------------------------------------------------
# C-11 CLI argparse: --transport and --port flags are declared on connect_p
# ---------------------------------------------------------------------------


def test_cli_connect_transport_flag_declared() -> None:
    """The 'connect' subparser must expose --transport and --port flags."""
    import argparse
    # Re-import to get a clean parser; we're just inspecting the parser structure
    # We run the parse logic without actually dispatching.
    import superlocalmemory.cli.main as slm_main

    # Build a parser the same way main() does but only run the argparse part
    # We do a minimal re-create by parsing known-good args
    sys.argv = ["slm", "connect", "cursor", "--transport", "http", "--port", "9000"]
    # Importing main and calling internal parse logic is fragile; use subprocess instead.
    import subprocess
    cp = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "connect", "--help"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "SLM_TEST_ISOLATION": "1"},
        cwd=str(_PROJECT_ROOT),
        timeout=10,
    )
    help_output = cp.stdout + cp.stderr
    assert "--transport" in help_output, f"--transport flag missing from connect help: {help_output[:500]}"
    assert "--port" in help_output, f"--port flag missing from connect help: {help_output[:500]}"


# ---------------------------------------------------------------------------
# C-12 connect_ide returns 'transport' key in result
# ---------------------------------------------------------------------------


def test_connect_ide_result_includes_transport_key(fake_home: Path) -> None:
    """Result dict must include 'transport' to aid debugging and --json output."""
    result = connect_ide("cursor", home=fake_home, transport="http")
    assert "transport" in result, "result dict missing 'transport' key"
    assert result["transport"] == "http"


def test_connect_ide_result_transport_default_is_stdio(fake_home: Path) -> None:
    """Default call (no transport arg) reports transport='stdio' in result."""
    result = connect_ide("cursor", home=fake_home)
    assert result.get("transport") == "stdio"


# ---------------------------------------------------------------------------
# C-13 http-mcp-remote + custom port
# ---------------------------------------------------------------------------


def test_transport_mcp_remote_custom_port(fake_home: Path) -> None:
    """http-mcp-remote with daemon_port=9001 should use port 9001 in args URL."""
    result = connect_ide("cursor", home=fake_home, transport="http-mcp-remote", daemon_port=9001)

    assert result["error"] is None
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]

    assert "9001" in block["args"][0]


# ---------------------------------------------------------------------------
# C-14 http-mcp-remote + profile appends to URL
# ---------------------------------------------------------------------------


def test_transport_mcp_remote_with_profile(fake_home: Path) -> None:
    """http-mcp-remote + profile='code' → URL in args has ?profile=code."""
    result = connect_ide(
        "cursor", home=fake_home, transport="http-mcp-remote", profile="code"
    )

    assert result["error"] is None
    desc = IDE_MATRIX["cursor"]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data[desc.server_key]["superlocalmemory"]

    assert "?profile=code" in block["args"][0], (
        f"Profile missing from mcp-remote URL: {block['args'][0]}"
    )
