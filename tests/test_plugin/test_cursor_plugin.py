# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""The Cursor-format plugin files that Grok Bot installs, as shipped.

Grok Bot runs Cursor-format plugins. On its plugin computer it spawns an MCP
server's ``command`` verbatim and does not replace ``${CLAUDE_PLUGIN_ROOT}``, so
the Claude Code ``.mcp.json`` (which names a script inside the plugin through
that variable) fails with ENOENT there. The Cursor manifest therefore points at
its own definition, ``plugin/mcp.cursor.json``, whose command is ``uvx`` on
PATH, pinned to this release.

These tests read the BUILT tree (what a user installs), not the generator.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin"
CURSOR_MANIFEST = PLUGIN / ".cursor-plugin" / "plugin.json"
CURSOR_MCP = PLUGIN / "mcp.cursor.json"
CURSOR_MARKETPLACE = REPO / ".cursor-plugin" / "marketplace.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_version() -> str:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def _server() -> dict:
    return _load(CURSOR_MCP)["mcpServers"]["superlocalmemory"]


def test_the_uvx_pin_is_the_package_version() -> None:
    server = _server()
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", f"superlocalmemory=={_package_version()}", "slm", "mcp"]


def test_manifest_and_marketplace_carry_the_package_version() -> None:
    assert _load(CURSOR_MANIFEST)["version"] == _package_version()
    assert _load(CURSOR_MARKETPLACE)["plugins"][0]["version"] == _package_version()


def test_no_cursor_file_contains_a_placeholder() -> None:
    """Every ``${VAR}`` in a Cursor listing must be declared as a user variable,
    and Grok Bot replaces none of them on its computer. So: none at all."""
    for path in (CURSOR_MANIFEST, CURSOR_MCP, CURSOR_MARKETPLACE):
        assert "${" not in path.read_text(encoding="utf-8"), path


def test_the_server_identifies_grok_bot_and_never_repoints_the_store() -> None:
    env = _server()["env"]
    assert env["SLM_AGENT_ID"] == "grok_bot"
    assert env["SLM_MCP_PROFILE"] == "core"
    assert env["SLM_NON_INTERACTIVE"] == "1"
    assert env["UV_TORCH_BACKEND"] == "cpu"
    assert "SLM_DATA_DIR" not in env


def test_manifest_paths_resolve_inside_the_plugin() -> None:
    manifest = _load(CURSOR_MANIFEST)
    for key in ("logo", "skills", "mcpServers"):
        rel = manifest[key]
        assert not rel.startswith("/") and ".." not in rel.split("/"), (key, rel)
        target = (PLUGIN / rel).resolve()
        assert target.exists(), (key, target)
        assert PLUGIN.resolve() in target.parents or target == PLUGIN.resolve()


def test_every_skill_has_the_frontmatter_cursor_requires() -> None:
    skills = sorted((PLUGIN / "skills").glob("*/SKILL.md"))
    assert skills
    for skill in skills:
        text = skill.read_text(encoding="utf-8")
        front = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        assert front, skill
        assert re.search(r"^name:\s*\S", front.group(1), re.M), skill
        assert re.search(r"^description:\s*\S", front.group(1), re.M), skill


def test_the_cursor_marketplace_offers_only_the_cursor_ready_tree() -> None:
    plugins = _load(CURSOR_MARKETPLACE)["plugins"]
    assert [(p["name"], p["source"]) for p in plugins] == [("superlocalmemory", "./plugin")]


def test_claude_code_still_gets_its_own_definition() -> None:
    """Adding the Cursor files must not change what Claude Code runs."""
    claude = _load(PLUGIN / ".claude-plugin" / "plugin.json")
    assert claude["mcpServers"] == "./.mcp.json"
    server = _load(PLUGIN / ".mcp.json")["mcpServers"]["superlocalmemory"]
    assert server["command"] == "${ComSpec:-${CLAUDE_PLUGIN_ROOT}/scripts/slm-launch}"
    assert server["env"] == {"SLM_AGENT_ID": "claude_code"}
