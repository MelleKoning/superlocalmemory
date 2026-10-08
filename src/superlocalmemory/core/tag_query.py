# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Exact tag filtering for ``search`` and ``list_recent`` (4.1.22).

Recall filters tags inside retrieval (``retrieval.facets`` plus the
search-inside supplement in ``retrieval.tag_search``). The two display paths,
full-text ``search`` and newest-first ``list_recent``, do not go through
retrieval, so they get the same filter here.

Membership comes first and the fetch is restricted to it in SQL. Filtering
after an ordinary fetch would lose a tag whose few memories sit below a
crowd of newer or better-matching untagged ones: the window would fill with
the crowd and the tagged memories would never be looked at. Membership is
``retrieval.tag_search.tag_members`` - the same all/any semantics, the same
canonical ``tag_key`` and the same personal scope that recall uses, so a tag
filter means the same thing on every surface.

A kind filter composes on top: ``core.kind_query`` runs its windowed kind
match over this restricted fetch instead of over the whole profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from superlocalmemory.storage.models import AtomicFact

FetchFn = Callable[[int], list["AtomicFact"]]


@dataclass(frozen=True, slots=True)
class TagFilter:
    """A cleaned ``tags`` / ``tags_match`` pair. ``tags`` empty = no filter."""

    tags: tuple[str, ...] = ()
    match: str = "all"

    @classmethod
    def of(cls, tags: object = None, tags_match: object = None) -> "TagFilter":
        from superlocalmemory.retrieval.facets import Facets

        facets = Facets.of(tags=tags, tags_match=tags_match)
        return cls(facets.tags, facets.tags_match)

    def __bool__(self) -> bool:
        return bool(self.tags)

    def facets(self) -> Any:
        from superlocalmemory.retrieval.facets import Facets

        return Facets.of(tags=list(self.tags), tags_match=self.match)


def members(db: Any, profile_id: str, tag_filter: TagFilter) -> frozenset[str]:
    """Visible fact ids of the profile that satisfy the filter.

    An unreadable membership keeps nothing, never everything: a tag filter
    is a hard filter, and ``tag_scope`` then says the check could not run.
    """
    from superlocalmemory.retrieval.tag_search import tag_members

    found = tag_members(db, profile_id, tag_filter.tags, tag_filter.match)
    return found if found is not None else frozenset()


def _ids_param(ids: frozenset[str]) -> str:
    return json.dumps(sorted(ids))


def recent_fetch(db: Any, profile_id: str, ids: frozenset[str]) -> FetchFn:
    """Newest-first fetch of ``limit`` facts among ``ids``."""

    def fetch(limit: int) -> list["AtomicFact"]:
        if not ids or limit <= 0:
            return []
        rows = db.execute(
            "SELECT * FROM atomic_facts WHERE profile_id = ? "
            "AND fact_id IN (SELECT value FROM json_each(?))"
            f"{db.visible_fact_clause()} ORDER BY created_at DESC LIMIT ?",
            (profile_id, _ids_param(ids), int(limit)),
        )
        return [db._row_to_fact(r) for r in rows]

    return fetch


def search_fetch(db: Any, query: str, profile_id: str, ids: frozenset[str]) -> FetchFn:
    """Full-text fetch of ``limit`` facts among ``ids``, best match first.

    Same MATCH expression and the same current-fact rule as
    ``DatabaseManager.search_facts_fts``; only the id restriction is added.
    """
    from superlocalmemory.storage.fts_terms import search_match_expression

    match_expr = search_match_expression(query)

    def fetch(limit: int) -> list["AtomicFact"]:
        if not ids or not match_expr or limit <= 0:
            return []
        rows = db.execute(
            "SELECT f.* FROM atomic_facts_fts AS fts "
            "JOIN atomic_facts AS f ON f.fact_id = fts.fact_id "
            "WHERE fts.atomic_facts_fts MATCH ? AND f.profile_id = ? "
            "AND f.fact_id IN (SELECT value FROM json_each(?)) "
            f"{db.current_fact_clause('f')} ORDER BY fts.rank LIMIT ?",
            (match_expr, profile_id, _ids_param(ids), int(limit)),
        )
        return [db._row_to_fact(r) for r in rows]

    return fetch


def report(db: Any, profile_id: str, tag_filter: TagFilter, matched: int) -> dict:
    """``tag_scope`` for a search or list response, worded as recall words it."""
    from superlocalmemory.retrieval.tag_scope import build_report

    return build_report(db, profile_id, tag_filter.facets(), matched)


def with_report(result: dict, db: Any, profile_id: str, tag_filter: TagFilter) -> dict:
    """``result`` plus ``tag_scope`` when a tag filter was asked for; a new dict."""
    if not tag_filter:
        return result
    return {**result, "tag_scope": report(db, profile_id, tag_filter,
                                          int(result.get("count", 0)))}


__all__ = ["TagFilter", "members", "recent_fetch", "report", "search_fetch", "with_report"]
