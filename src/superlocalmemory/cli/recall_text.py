# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Plain-text lines for ``slm recall`` that the JSON already carries.

An incomplete recall (a channel abandoned at the hang guard, an embedding model
still loading on a just-started daemon, or the daemon's over-budget keyword
fallback) used to print exactly what a complete one prints. With no results
that was "No confident match." — a confident statement about the store, made
by a search that did not look everywhere. These lines say what was skipped.
"""

from __future__ import annotations

__all__ = ["empty_result_line", "incomplete_line"]


def incomplete_line(result: dict) -> str:
    """One line naming the channels this answer ran without, or ""."""
    skipped = [str(c) for c in (result.get("incomplete_channels") or [])]
    if not skipped:
        return ""
    status = result.get("channel_status") or {}
    warming = any(status.get(c) == "warming" for c in skipped)
    reason = (
        "the embedding model is still loading"
        if warming else "part of the search did not finish"
    )
    return (
        f"Incomplete search: {reason}; searched without "
        f"{', '.join(sorted(skipped))}. Ask again in a moment for a full answer."
    )


def empty_result_line(result: dict) -> str:
    """What to print when no memory came back."""
    if incomplete_line(result):
        return "Nothing found yet."
    if result.get("no_confident_match"):
        return "No confident match."
    return "No matching memories found."
