# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Shared direct-engine reads that take a ``kind`` filter (LLD/WP8 4.1.19).

``list_recent`` (MCP) and ``list`` (CLI) both read straight off
``DatabaseManager.get_all_facts`` — no daemon round trip — and ``search``
(MCP) reads off ``DatabaseManager.search_facts_fts`` the same way. Both
surfaces need the identical over-fetch + filter-by-displayed-kind behaviour,
so it lives here once and each surface imports it rather than writing its own
loop that could drift from the other's.

``recall`` and HTTP ``/recall`` do NOT use this module — their retrieval goes
through ``RetrievalEngine.recall`` and the shared serialization chokepoint
(``server.recall_serializer.serialize_recall_response``), which applies the
same ``kind_filter`` primitives at that different layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from superlocalmemory.retrieval.kind_filter import filter_items_by_kind, overfetch_limit
from superlocalmemory.storage.memory_kinds import MemoryKind, kind_fields, parse_kind

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import AtomicFact


class InvalidKind(ValueError):
    """Raised when a caller-supplied kind string does not parse (I1 — never
    lets a bad filter reach retrieval; the caller decides how to report it)."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        super().__init__(
            "Unknown memory kind. Use one of: " + ", ".join(k.value for k in MemoryKind)
        )


def resolve_kind(raw: str | None) -> str | None:
    """Parse a caller-supplied kind filter string.

    Empty or whitespace-only input means "no filter" (``None``). Anything
    else must parse via ``storage.memory_kinds.parse_kind`` (canonical value
    or a known alias) or ``InvalidKind`` is raised — every surface refuses
    before any retrieval happens, never silently ignores a typo.
    """
    text = (raw or "").strip()
    if not text:
        return None
    parsed = parse_kind(text)
    if parsed is None:
        raise InvalidKind(text)
    return parsed.value


def _kind_matching_facts(facts: list["AtomicFact"], kind: str | None,
                         limit: int) -> list["AtomicFact"]:
    if not kind:
        return facts[:limit]
    kept: list["AtomicFact"] = []
    for fact in facts:
        if kind_fields(fact)["memory_kind"] == kind:
            kept.append(fact)
            if len(kept) >= limit:
                break
    return kept


def list_recent_facts(
    db: "DatabaseManager", profile_id: str, limit: int, kind: str | None,
) -> list["AtomicFact"]:
    """``get_all_facts``, newest first, with an optional displayed-kind filter.

    When ``kind`` is set, over-fetches (capped) so filtering still leaves
    room for up to ``limit`` matches; when it is not, behaviour is
    byte-identical to calling ``get_all_facts`` directly.
    """
    fetch_n = overfetch_limit(limit) if kind else limit
    facts = db.get_all_facts(profile_id, limit=fetch_n)
    return _kind_matching_facts(facts, kind, limit)


def search_facts(
    db: "DatabaseManager", query: str, profile_id: str, limit: int, kind: str | None,
) -> list["AtomicFact"]:
    """``search_facts_fts`` with an optional displayed-kind filter.

    Same over-fetch contract as ``list_recent_facts``.
    """
    fetch_n = overfetch_limit(limit) if kind else limit
    facts = db.search_facts_fts(query, profile_id, limit=fetch_n)
    return _kind_matching_facts(facts, kind, limit)


def kind_item(fact: "AtomicFact") -> dict[str, Any]:
    """The five ``kind_fields`` for one fact, for merging into a result dict."""
    return kind_fields(fact)


__all__ = [
    "InvalidKind", "kind_item", "list_recent_facts", "resolve_kind", "search_facts",
]
