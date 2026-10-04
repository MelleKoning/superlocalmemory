# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The number in a sentence is a promise, and prose goes stale more quietly
than code.

A sibling test compares the count claimed in the packaging JSON against the
profile it names. It reads three JSON files, so every claim written in
markdown drifted unchecked: the README announced a 16-tool surface for a
plugin that registers 34, and the rules file shipped to the agent announced 16
tools above a table listing 14, for a profile holding 18.

This reads the markdown a user reads, and the table an agent reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from superlocalmemory.mcp.profiles import _PROFILE_ALIASES, _PROFILE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]

# Prose a human or an agent reads before deciding what this thing can do.
PROSE = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "plugin-src" / "rules" / "AGENTS.md",
    # Hand-maintained, deliberately not a copy of the Claude one — which is
    # exactly why it drifted two releases further behind than the others.
    REPO_ROOT / "codex-plugin" / "AGENTS.md",
    REPO_ROOT / "docs" / "mcp-tools.md",
    REPO_ROOT / "codex-plugin" / "README.md",
    REPO_ROOT / "plugin-src" / "MCP_JSON_NOTES.md",
)

# Rules documents that describe the core surface tool by tool.
TOOL_TABLES = tuple(p for p in PROSE if p.name == "AGENTS.md")

_NAMES = "|".join(sorted(set(_PROFILE_DEFINITIONS) | set(_PROFILE_ALIASES)))

# Every shape a count claim is written in today. Each captures (count, name)
# or (name, count) — the group names say which.
_CLAIM_SHAPES = (
    # "34-tool code profile", "18-tool `core` memory surface"
    re.compile(rf"(?P<count>\d+)-tool `?(?P<name>{_NAMES})`?\b"),
    # "Tool reference (core profile — 18 tools)"
    re.compile(rf"\b(?P<name>{_NAMES})`? profile[^)\n]*?(?P<count>\d+) tools"),
    # "Use `full` (50 tools)"
    re.compile(rf"`(?P<name>{_NAMES})` \((?P<count>\d+) tools\)"),
    # A table row whose second column is the count: "| `core` | 18 | ..." and
    # "| `full` (and unset) | 54 |". The rewritten README states its counts
    # only in such a table, which no shape above read.
    re.compile(rf"^\| `(?P<name>{_NAMES})`[^|`\n]*\| (?P<count>\d+) \|", re.MULTILINE),
)


def _claims(text: str) -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for shape in _CLAIM_SHAPES:
        for m in shape.finditer(text):
            found.append((m.group("name"), int(m.group("count")), m.group(0)))
    return found


@pytest.mark.parametrize("path", PROSE, ids=lambda p: p.name)
def test_a_count_written_in_prose_matches_the_profile(path: Path) -> None:
    assert path.exists(), f"{path} is gone; update this test or restore the file"

    claims = _claims(path.read_text(encoding="utf-8"))
    assert claims, (
        f"{path.name} no longer states any tool count; if that was deliberate, "
        f"delete this test rather than leaving it passing vacuously"
    )

    for raw_name, advertised, quote in claims:
        name = _PROFILE_ALIASES.get(raw_name, raw_name)
        real = len(_PROFILE_DEFINITIONS[name])
        assert advertised == real, (
            f"{path.name} says {quote!r}; the {name!r} profile holds {real} tools"
        )


@pytest.mark.parametrize(
    "path", TOOL_TABLES, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_the_table_an_agent_reads_lists_every_core_tool(path: Path) -> None:
    """A count can be right while the list below it is short.

    The rules file is the only description of the surface an agent sees before
    it calls anything. A tool missing from that table is a tool the agent never
    learns exists, whatever the heading claims.
    """
    text = path.read_text(encoding="utf-8")
    section = text.split("## Tool reference", 1)
    assert len(section) == 2, f"{path} no longer has a '## Tool reference' section"
    table = section[1].split("\n## ", 1)[0]

    listed = set(re.findall(r"^\| `([a-z_]+)`", table, flags=re.MULTILINE))
    core = _PROFILE_DEFINITIONS["core"]

    assert listed == core, (
        f"tool table diff — listed but not in core: {sorted(listed - core)}, "
        f"in core but never described: {sorted(core - listed)}"
    )
