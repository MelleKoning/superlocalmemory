# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Validation of MCP tool-profile names (``SLM_MCP_PROFILE``, ``--profile``).

Profile names are a closed set: the canonical names, ``whole`` for every
tool, and the count-suffixed aliases earlier releases published. Anything
else is refused (fail closed, so a typo never widens the tool surface), but
refused with a sentence that lists the valid names rather than a traceback.
"""

from __future__ import annotations

import os
import sys

from superlocalmemory.mcp.profiles import _PROFILE_ALIASES, _PROFILE_DEFINITIONS

WHOLE = "whole"


class UnknownProfileError(ValueError):
    """A profile name that is neither canonical nor a published alias."""


def valid_profile_names() -> tuple[str, ...]:
    """Canonical names in documentation order, ending with ``whole``."""
    return (*_PROFILE_DEFINITIONS, WHOLE)


def canonical_profile(name: str) -> str:
    """Return the canonical profile for *name* ("" means no selection).

    Raises:
        UnknownProfileError: for any name SLM does not define.
    """
    wanted = (name or "").strip().lower()
    canonical = _PROFILE_ALIASES.get(wanted, wanted)
    if not canonical or canonical == WHOLE or canonical in _PROFILE_DEFINITIONS:
        return canonical
    raise UnknownProfileError(unknown_profile_message(name))


def unknown_profile_message(name: str, source: str = "SLM_MCP_PROFILE") -> str:
    valid = ", ".join(valid_profile_names())
    return (
        f"{source}={name!r} is not recognised; valid profiles: {valid} "
        f"({WHOLE} registers every tool)."
    )


def exit_on_unknown_profile() -> None:
    """Refuse to start ``slm mcp`` with an unknown ``SLM_MCP_PROFILE``.

    Writes to stderr only: stdout is the MCP JSON-RPC channel.
    """
    raw = os.environ.get("SLM_MCP_PROFILE", "")
    try:
        canonical_profile(raw)
    except UnknownProfileError as exc:
        print(f"slm mcp: {exc}", file=sys.stderr)
        sys.exit(2)
