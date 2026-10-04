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

__all__ = ["empty_result_line", "incomplete_line", "remember_receipt_text"]


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


def remember_receipt_text(result: dict) -> str:
    """The ``slm remember`` receipt, saying which searches can reach it yet.

    "Queryable" is true from the moment of admission for search by the
    memory's own words (the full-text index is written in the same
    transaction). Meaning-based search needs the memory's vector, which the
    background indexer adds a moment later — so a paraphrased question can
    miss it until then, and the receipt says so instead of implying more.
    """
    state = str(result.get("materialization_state") or "queryable")
    if state == "accepted":
        return (
            "Saved ✓ (durable). The memory writer is busy, so it is being "
            "indexed now and will be searchable within seconds "
            f"(admission={result.get('admission_id', 'unknown')})."
        )
    line = (f"{state.capitalize()} ✓ {result.get('count', 0)} facts "
            f"(operation={result.get('operation_id', 'unknown')}).")
    if state in ("queryable", "enriching"):
        line += ("\nFindable now by its words; meaning-based search catches "
                 "up when background indexing finishes (usually seconds).")
    return line
