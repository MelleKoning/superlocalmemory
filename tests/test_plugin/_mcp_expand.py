# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Expand a plugin `.mcp.json` entry the way Claude Code does (#139).

Claude Code (checked in the 2.1.229 binary) handles a plugin MCP string in two
passes: first a literal replacement of ``${CLAUDE_PLUGIN_ROOT}`` (and
``${CLAUDE_PLUGIN_DATA}``, ``${CLAUDE_PROJECT_DIR}``), then environment
expansion with this exact pattern:

    /\\$\\{([A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?)\\}/g

``${VAR}`` takes the environment value; ``${VAR:-default}`` takes the default
when VAR is unset; an unset VAR without a default is left as written and
reported missing. The docs describe the same two forms:
https://code.claude.com/docs/en/mcp (environment variable expansion).

Because the plugin root is substituted first, ``${ComSpec:-${CLAUDE_PLUGIN_ROOT}/x}``
is a single, un-nested ``${VAR:-default}`` by the time the second pass sees it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Mapping

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?)\}")

#: Windows always defines this (the command interpreter); POSIX does not.
WINDOWS_ENV = {"ComSpec": r"C:\WINDOWS\system32\cmd.exe"}


def expand(value: str, plugin_root: str, lookup: Callable[[str], str | None]) -> tuple[str, list[str]]:
    missing: list[str] = []
    value = value.replace("${CLAUDE_PLUGIN_ROOT}", plugin_root)

    def _sub(match: re.Match[str]) -> str:
        body = match.group(1)
        name, sep, default = body.partition(":-")
        found = lookup(name)
        if isinstance(found, str):
            return found
        if sep:
            return default
        missing.append(name)
        return match.group(0)

    return _ENV_PATTERN.sub(_sub, value), missing


def declared_server(mcp_json: Path) -> dict:
    data = json.loads(mcp_json.read_text(encoding="utf-8"))
    return data["mcpServers"]["superlocalmemory"]


def expanded_argv(mcp_json: Path, plugin_root: str, env: Mapping[str, str]) -> list[str]:
    """The argv the host would spawn, or AssertionError if anything is unresolved."""
    server = declared_server(mcp_json)
    lookup = env.get
    out: list[str] = []
    for raw in [server["command"], *server.get("args", [])]:
        value, missing = expand(raw, plugin_root, lookup)
        assert not missing, f"{mcp_json}: unresolved {missing} in {raw!r}"
        out.append(value)
    return out
