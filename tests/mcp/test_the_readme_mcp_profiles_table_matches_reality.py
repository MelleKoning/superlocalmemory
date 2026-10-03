# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""README.md's "### MCP Profiles" table is a markdown table, not prose, so
none of the sibling count-checking tests parse it — it drifted to
core 16 / code 31 / mesh 8 / full 49 / power 61 / whole 94 (real:
18 / 38 / 8 / 54 / 66 / 101) with nothing catching it.
"""

from __future__ import annotations

import re
from pathlib import Path

from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

# "whole" is deliberately absent from `_PROFILE_DEFINITIONS` (it means the
# raw server, all tools). Pinned by
# tests/test_mcp/test_mcp_exposure_contract.py
# (`test_registration_exposure_is_exact_and_duplicate_free`, exposure
# "whole", expected_count 101).
_WHOLE_TOOLS_COUNT = 101

_ROW = re.compile(r"^\| `(\w+)` \| (\d+) \|", flags=re.MULTILINE)


def _table_text() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("### MCP Profiles", 1)
    assert len(section) == 2, "README.md no longer has a '### MCP Profiles' section"
    # Table ends at the next heading.
    return section[1].split("\n#", 1)[0]


def test_every_row_in_the_mcp_profiles_table_matches_the_real_count() -> None:
    table = _table_text()
    rows = _ROW.findall(table)
    assert rows, (
        "README.md's MCP Profiles table no longer states a tool count per "
        "profile; if that was deliberate, delete this test rather than "
        "leaving it passing vacuously"
    )

    for name, advertised in rows:
        real = _WHOLE_TOOLS_COUNT if name == "whole" else len(_PROFILE_DEFINITIONS[name])
        assert int(advertised) == real, (
            f"README.md's MCP Profiles table says {name!r} is {advertised} "
            f"tools; it actually holds {real}"
        )


def test_every_named_profile_plus_whole_appears_in_the_table() -> None:
    table = _table_text()
    named = {name for name, _ in _ROW.findall(table)}
    expected = set(_PROFILE_DEFINITIONS) | {"whole"}
    assert named == expected, (
        f"table lists {sorted(named)}, expected {sorted(expected)}"
    )
