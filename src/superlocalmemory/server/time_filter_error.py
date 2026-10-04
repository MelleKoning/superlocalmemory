# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One HTTP answer for an unreadable recall window, on every route (R5).

``{"error": "invalid_window", "code": "INVALID_TIME_FILTER", "field": "window",
"message": ...}`` with status 400 — the ``error`` key matches the existing
``invalid_as_of`` replies, ``code`` matches the CLI's and MCP's.
"""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse

from superlocalmemory.retrieval.time_filter import InvalidTimeFilter, check_window


def invalid_time_filter_response(exc: InvalidTimeFilter) -> JSONResponse:
    """The one 400 body every HTTP recall route returns for an unreadable filter."""
    return JSONResponse({"error": f"invalid_{exc.field}", **exc.as_dict()},
                        status_code=400)


def checked_window_or_400(window: Any) -> str | JSONResponse:
    """The window as a string ("" for none), or the 400 to return instead."""
    try:
        checked = check_window(window)
    except InvalidTimeFilter as exc:
        return invalid_time_filter_response(exc)
    return checked or ""


__all__ = ["checked_window_or_400", "invalid_time_filter_response"]
