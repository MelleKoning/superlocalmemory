# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The dashboard's "connect your IDE" panel quotes a tool count per profile
in a static JS string (``od-mcp.js``) — the one count a user reads before
picking a profile. It drifted to core 14 / code 28 / full 46 / power 58
(real: 18 / 38 / 54 / 66) because nothing compared the string to the real
profile sets. Sibling tests already do this for manifest.json/plugin.json
(test_the_advertised_tool_count_is_the_real_one.py) and for README/AGENTS.md
(test_the_prose_a_user_reads_states_the_real_tool_count.py); this is the same
check for the dashboard's own copy, so none of the three can drift again
without a test catching it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
OD_MCP_JS = REPO_ROOT / "src" / "superlocalmemory" / "ui" / "js" / "od-mcp.js"

# '  "SLM_MCP_PROFILE": "core"    // 18 tools — minimal\n' +
_CLAIM = re.compile(r'"SLM_MCP_PROFILE":\s*"(\w+)"\s*//\s*(\d+)\s*tools')


def test_the_dashboard_setup_snippet_matches_the_real_profile_counts() -> None:
    assert OD_MCP_JS.exists(), f"{OD_MCP_JS} is gone; update this test or restore the file"

    text = OD_MCP_JS.read_text(encoding="utf-8")
    claims = _CLAIM.findall(text)
    assert claims, (
        f"{OD_MCP_JS.name} no longer states a tool count per profile; if that was "
        f"deliberate, delete this test rather than leaving it passing vacuously"
    )

    for name, advertised in claims:
        assert name in _PROFILE_DEFINITIONS, f"{OD_MCP_JS.name} names unknown profile {name!r}"
        real = len(_PROFILE_DEFINITIONS[name])
        assert int(advertised) == real, (
            f"{OD_MCP_JS.name} says {name!r} is {advertised} tools; it actually holds {real}"
        )


def test_every_named_profile_is_quoted_in_the_setup_snippet() -> None:
    """A count can be right while a whole profile silently drops out of the
    snippet - catch that too, not just a wrong number for the ones present."""
    text = OD_MCP_JS.read_text(encoding="utf-8")
    quoted = {name for name, _ in _CLAIM.findall(text)}
    assert quoted == set(_PROFILE_DEFINITIONS), (
        f"snippet lists {sorted(quoted)}, real profiles are {sorted(_PROFILE_DEFINITIONS)}"
    )


# The file-header example response (an illustration of what GET
# /api/v3/mcp/profiles returns), e.g.:
#   core:  { count: 14, tools: [...], description: "..." },
_HEADER_EXAMPLE_CLAIM = re.compile(r"(\w+):\s*\{\s*count:\s*(\d+),")


def test_the_header_example_response_matches_the_real_profile_counts() -> None:
    """od-mcp.js opens with a worked example of the endpoint's JSON shape.
    It is documentation, not live code, so nothing re-derives it from the
    real profile sets — it drifted to core 14 / code 28 / full 46 / power 58
    while the live snippet below it (checked above) was already correct.
    """
    text = OD_MCP_JS.read_text(encoding="utf-8")
    header = text.split("// CSP-safe", 1)[0]
    claims = [
        (name, int(count))
        for name, count in _HEADER_EXAMPLE_CLAIM.findall(header)
        if name in _PROFILE_DEFINITIONS
    ]
    assert claims, (
        f"{OD_MCP_JS.name}'s header example no longer states a tool count per "
        f"profile; if that was deliberate, delete this test rather than "
        f"leaving it passing vacuously"
    )
    for name, advertised in claims:
        real = len(_PROFILE_DEFINITIONS[name])
        assert advertised == real, (
            f"{OD_MCP_JS.name} header example says {name!r} is {advertised} "
            f"tools; it actually holds {real}"
        )
