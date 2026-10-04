# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One check for recall's time filters, used by every door (4.1.20, R5).

An unreadable ``window`` used to be passed straight through by MCP and HTTP;
the engine could not parse it and applied no filter at all, so a caller who
asked for "last week" with a typo got every memory back and was told nothing.
The CLI already refused it. Now the CLI, MCP, every HTTP route and the engine
itself ask this module, and all say the same thing: ``INVALID_TIME_FILTER``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from superlocalmemory.retrieval.time_window import parse_window

CODE = "INVALID_TIME_FILTER"

_WINDOW_EXPECTED = ("Expected a relative span (24h, 7d, 30d, 1y) or a range "
                    "(2026-07-01..2026-07-31).")
_INSTANT_EXPECTED = "Expected ISO 8601 UTC datetime, e.g. '2024-01-01T00:00:00Z'."


class InvalidTimeFilter(ValueError):
    """A time filter was supplied and cannot be read. Never silently dropped."""

    code = CODE

    def __init__(self, field: str, raw: Any) -> None:
        self.field = field
        self.raw = raw
        expected = _WINDOW_EXPECTED if field == "window" else _INSTANT_EXPECTED
        super().__init__(f"invalid {field} value: {raw!r}. {expected}")

    def as_dict(self) -> dict[str, str]:
        return {"code": CODE, "field": self.field, "message": str(self)}


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (tuple, list)):
        return len(value) == 0
    return False


def check_window(window: Any) -> Any:
    """The window to apply, or None for "no filter". Raises if unreadable.

    Blank means absent. Anything else must parse; a string is returned
    stripped, a ``(start, end)`` pair unchanged.
    """
    if _blank(window):
        return None
    if isinstance(window, str):
        window = window.strip()
    elif not isinstance(window, (tuple, list)):
        raise InvalidTimeFilter("window", window)
    if parse_window(window) is None:
        raise InvalidTimeFilter("window", window)
    return window


def first_invalid(filters: Mapping[str, Any]) -> InvalidTimeFilter | None:
    """The first unreadable filter among ``{field: raw}``, or None.

    ``window`` is checked as a window; every other field as an instant
    (``as_of``, ``known_as_of``, ``valid_at``).
    """
    from superlocalmemory.retrieval.temporal_utils import normalize_as_of

    for field, raw in filters.items():
        if _blank(raw):
            continue
        if field == "window":
            try:
                check_window(raw)
            except InvalidTimeFilter as exc:
                return exc
        elif not isinstance(raw, str) or normalize_as_of(raw.strip()) is None:
            return InvalidTimeFilter(field, raw)
    return None


__all__ = ["CODE", "InvalidTimeFilter", "check_window", "first_invalid"]
