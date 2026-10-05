# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""What a saved view is, and what a valid one looks like (issue #113).

A saved view is a NAMED RECALL QUERY: a name, the query text, a small set of
recall filters and how many results to show. Nothing else. There is no prompt
and no model run over the store, so a view is as traceable and as repeatable
as the recall it stands for, and it is not an injection surface: the query
text goes to recall exactly as if the person had typed it.

Every value is validated here, once, before it is stored or run. Each surface
(CLI, MCP, HTTP, dashboard) reports the same refusal with the same code.

FILTERS ARE A REGISTRY, NOT A FIXED SET
---------------------------------------
:data:`FILTERS` maps a filter name to the function that validates it and
returns the normalized value recall takes. A filter recall gains later (a
project filter, say) is one entry here plus the line that passes it on in
``views.runner`` — no table change, because filters are stored as JSON.
An unknown filter is refused, never silently dropped: a view that quietly
ignores part of what it was told would show a different answer than the one
its owner saved.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MAX_NAME_CHARS = 80
MAX_QUERY_CHARS = 1000
MAX_VIEWS_PER_PROFILE = 200
DEFAULT_LIMIT = 10
MAX_LIMIT = 50

# Stable refusal codes. Surfaces map them to HTTP statuses and exit codes.
INVALID_NAME = "invalid_view_name"
INVALID_QUERY = "invalid_view_query"
INVALID_LIMIT = "invalid_view_limit"
INVALID_FILTER = "invalid_view_filter"
UNKNOWN_FILTER = "unknown_view_filter"
VIEW_EXISTS = "view_exists"
VIEW_NOT_FOUND = "view_not_found"
TOO_MANY_VIEWS = "too_many_views"
VIEWS_UNAVAILABLE = "views_unavailable"

#: Codes that mean "the request is wrong" (HTTP 422, CLI exit 2).
INPUT_CODES = frozenset({INVALID_NAME, INVALID_QUERY, INVALID_LIMIT,
                         INVALID_FILTER, UNKNOWN_FILTER})


class ViewError(ValueError):
    """A refused view operation. ``code`` is stable; ``message`` is plain English."""

    def __init__(self, code: str, message: str, *, field: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field

    def as_dict(self) -> dict[str, str]:
        out = {"code": self.code, "message": self.message}
        if self.field:
            out["field"] = self.field
        return out


def _has_control_chars(text: str, *, allow: str = "") -> bool:
    return any(unicodedata.category(ch) in ("Cc", "Cf") and ch not in allow
               for ch in text)


def validate_name(raw: Any) -> str:
    """A view name: 1-80 visible characters, no line breaks or control codes."""
    if not isinstance(raw, str):
        raise ViewError(INVALID_NAME, "A view name must be text.", field="name")
    name = " ".join(raw.split())
    if not name:
        raise ViewError(INVALID_NAME, "Give the view a name.", field="name")
    if len(name) > MAX_NAME_CHARS:
        raise ViewError(INVALID_NAME,
                        f"A view name can be at most {MAX_NAME_CHARS} characters.",
                        field="name")
    if _has_control_chars(name):
        raise ViewError(INVALID_NAME, "A view name cannot contain control characters.",
                        field="name")
    return name


def name_key(name: str) -> str:
    """The form two names are compared in: "Work log" and "work LOG" are one name."""
    return unicodedata.normalize("NFKC", name).casefold()


def validate_query(raw: Any) -> str:
    """The recall query: 1-1000 characters of text."""
    if not isinstance(raw, str):
        raise ViewError(INVALID_QUERY, "The query must be text.", field="query")
    query = raw.strip()
    if not query:
        raise ViewError(INVALID_QUERY, "Say what the view should look for.", field="query")
    if len(query) > MAX_QUERY_CHARS:
        raise ViewError(INVALID_QUERY,
                        f"A view query can be at most {MAX_QUERY_CHARS} characters.",
                        field="query")
    if _has_control_chars(query, allow="\n\t"):
        raise ViewError(INVALID_QUERY, "The query cannot contain control characters.",
                        field="query")
    return query


def validate_limit(raw: Any) -> int:
    """How many results the view shows: a whole number from 1 to 50."""
    if raw is None:
        return DEFAULT_LIMIT
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= MAX_LIMIT:
        raise ViewError(INVALID_LIMIT,
                        f"The number of results must be a whole number from 1 to {MAX_LIMIT}.",
                        field="limit")
    return raw


# -- filters ------------------------------------------------------------------


def _text(name: str, raw: Any) -> str:
    if not isinstance(raw, str):
        raise ViewError(INVALID_FILTER, f"The {name} filter must be text.", field=name)
    return raw.strip()


def _kind(raw: Any) -> str | None:
    from superlocalmemory.core.kind_query import InvalidKind, resolve_kind

    try:
        return resolve_kind(_text("kind", raw))
    except InvalidKind as exc:
        raise ViewError(INVALID_FILTER, str(exc), field="kind") from exc


def _window(raw: Any) -> str | None:
    from superlocalmemory.retrieval.time_filter import InvalidTimeFilter, check_window

    try:
        return check_window(_text("window", raw)) or None
    except InvalidTimeFilter as exc:
        raise ViewError(INVALID_FILTER, str(exc), field="window") from exc


def _as_of(raw: Any) -> str | None:
    from superlocalmemory.retrieval.temporal_utils import normalize_as_of

    text = _text("as_of", raw)
    if not text:
        return None
    normalized = normalize_as_of(text)
    if normalized is None:
        raise ViewError(INVALID_FILTER,
                        f"invalid as_of value: {text!r}. Use an ISO-8601 date and time, "
                        "e.g. 2026-01-01T00:00:00Z.", field="as_of")
    return normalized


#: Filter name -> validator returning the normalized value, or None for "unset".
#: Each is a filter recall already takes under the same name.
FILTERS: Mapping[str, Callable[[Any], str | None]] = MappingProxyType({
    "kind": _kind,
    "window": _window,
    "as_of": _as_of,
})


def validate_filters(raw: Any) -> tuple[tuple[str, str], ...]:
    """Normalized ``(name, value)`` pairs, sorted by name; blank values dropped."""
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ViewError(INVALID_FILTER, "Filters must be a set of name/value pairs.",
                        field="filters")
    unknown = sorted(str(k) for k in raw if k not in FILTERS)
    if unknown:
        raise ViewError(
            UNKNOWN_FILTER,
            f"Unknown filter {', '.join(unknown)}. A view can filter by: "
            f"{', '.join(sorted(FILTERS))}.", field="filters")
    pairs = []
    for name in sorted(raw):
        value = FILTERS[name](raw[name])
        if value:
            pairs.append((name, value))
    return tuple(pairs)


# -- the view -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SavedView:
    """One saved view. Immutable; a rename produces a new instance."""

    view_id: str
    profile_id: str
    name: str
    query: str
    filters: tuple[tuple[str, str], ...]
    limit: int
    created_at: str
    updated_at: str

    @property
    def filter_map(self) -> dict[str, str]:
        return dict(self.filters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "profile_id": self.profile_id,
            "name": self.name,
            "query": self.query,
            "filters": self.filter_map,
            "limit": self.limit,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


__all__ = [
    "DEFAULT_LIMIT", "FILTERS", "INPUT_CODES", "MAX_LIMIT", "MAX_NAME_CHARS",
    "MAX_QUERY_CHARS", "MAX_VIEWS_PER_PROFILE", "SavedView", "ViewError",
    "name_key", "validate_filters", "validate_limit", "validate_name",
    "validate_query",
    "INVALID_FILTER", "INVALID_LIMIT", "INVALID_NAME", "INVALID_QUERY",
    "TOO_MANY_VIEWS", "UNKNOWN_FILTER", "VIEWS_UNAVAILABLE", "VIEW_EXISTS",
    "VIEW_NOT_FOUND",
]
