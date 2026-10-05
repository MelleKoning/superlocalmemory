# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Running a saved view: hand it to recall, unchanged, and show what came back.

A view adds nothing to recall and takes nothing away. :func:`recall_arguments`
is the one translation from a stored view to recall's own parameters, used by
every surface, so the CLI, the dashboard and an agent all ask recall the same
question for the same view — and get the same ranking, the same answer check
and the same order on an unchanged store.

:func:`shape_run` keeps recall's answer as recall gave it: the same order, the
same scores, the confidence signals, and for every result the id of the memory
it came from (``fact_id`` — the id ``fetch``, ``slm delete`` and the dashboard
use). Nothing is summarised or re-ranked here.
"""

from __future__ import annotations

from typing import Any

from superlocalmemory.views.model import SavedView

#: Recall response fields a view run passes through untouched, when present.
_PASSTHROUGH = (
    "query_type", "no_confident_match", "abstained", "abstention_reason",
    "answer_confidence", "answer_check", "answer_check_status", "answer_check_note",
    "answer_check_ran", "answer_check_reason", "incomplete_channels", "query_id",
    "retrieval_mode", "retrieval_time_ms", "kind_filter_truncated",
)

#: Per-result fields shown for a view. Content is recall's own (already
#: budgeted) text; nothing here adds or rewrites any of it.
_RESULT_FIELDS = (
    "fact_id", "memory_id", "content", "score", "confidence", "created_at",
    "age_label", "fact_type", "memory_kind", "memory_kind_label", "memory_kind_state",
)


def recall_arguments(view: SavedView) -> dict[str, Any]:
    """Recall's keyword arguments for ``view``. Only set filters appear.

    Keys use recall's own names (``query``, ``limit``, ``window``, ``as_of``,
    ``kind``), so this dict reads the same on every surface that runs a view.
    """
    args: dict[str, Any] = {"query": view.query, "limit": view.limit}
    filters = view.filter_map
    for name in ("window", "as_of", "kind"):
        if filters.get(name):
            args[name] = filters[name]
    return args


def view_session_id(view: SavedView) -> str:
    """The recall session a view runs under: synthetic, so continuity ignores it.

    Minted through ``core.session_identity`` so the prefix is one the
    continuity guard knows. A plain id here would read as a real conversation:
    each run would take a working-set slot, and the second run of a view would
    be biased toward what the first one showed.
    """
    from superlocalmemory.core.session_identity import synthetic_session_id

    return synthetic_session_id("view", view.view_id)


def shape_run(view: SavedView, response: dict[str, Any]) -> dict[str, Any]:
    """A view run: the view, then recall's results in recall's order, with ids."""
    rows = []
    for rank, item in enumerate(response.get("results") or [], start=1):
        if not isinstance(item, dict):
            continue
        row = {key: item[key] for key in _RESULT_FIELDS if key in item}
        row["rank"] = rank
        rows.append(row)
    out: dict[str, Any] = {
        "success": True,
        "view": view.to_dict(),
        "profile": response.get("profile") or view.profile_id,
        "results": rows,
        "count": len(rows),
        "result_ids": [row.get("fact_id", "") for row in rows],
    }
    for key in _PASSTHROUGH:
        if key in response:
            out[key] = response[key]
    return out


__all__ = ["recall_arguments", "shape_run", "view_session_id"]
