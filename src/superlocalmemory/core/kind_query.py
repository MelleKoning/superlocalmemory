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
through ``RetrievalEngine.recall``, which applies its own kind filtering
earlier, in the retrieval layer, before any of this module's callers ever
see the candidates. ``server.recall_serializer`` only attaches the display
fields (``kind_fields``); it does not filter.
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


def engine_display_min_confidence(engine: Any) -> float:
    """The live ``memory_kinds.display_min_confidence`` off ``engine``, or
    the module default when ``engine`` has no ``_config`` (a minimal test
    stand-in). Centralised so every surface (MCP, CLI) reads the SAME
    configured threshold ``list_recent_facts``/``search_facts`` apply,
    rather than each guessing at its own fallback (4.1.19 M3).
    """
    config = getattr(engine, "_config", None)
    memory_kinds = getattr(config, "memory_kinds", None)
    value = getattr(memory_kinds, "display_min_confidence", None)
    return value if isinstance(value, (int, float)) else _DEFAULT_DISPLAY_MIN_CONFIDENCE


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


#: Matches storage.memory_kinds.kind_fields' own default, so a caller that
#: passes neither value agrees with itself. Every real caller that HAS a
#: live SLMConfig should pass config.memory_kinds.display_min_confidence
#: (4.1.19 M3 — this used to be silently hard-coded here too).
_DEFAULT_DISPLAY_MIN_CONFIDENCE = 0.20

#: 4.1.19 L2-13 / Muse M2: how far ``_windowed_kind_fetch`` will grow its
#: fetch window chasing ``limit`` matches before giving up and reporting
#: truncation. Bounds the extra work a kind filter can cost on a huge store;
#: well past this, a caller is better served by a narrower query than by the
#: direct-engine read this module exists for (see module docstring).
WINDOWED_FETCH_HARD_CAP = 2000

#: How fast the window grows between fetches. 4x keeps the number of
#: round trips small (an initial window of <=100 reaches the 2000 cap in
#: 3-4 fetches) regardless of how large the store is.
_WINDOW_GROWTH_FACTOR = 4


def _kind_matching_facts(
    facts: list["AtomicFact"], kind: str | None, limit: int, *,
    display_min_confidence: float = _DEFAULT_DISPLAY_MIN_CONFIDENCE,
) -> list["AtomicFact"]:
    if not kind:
        return facts[:limit]
    kept: list["AtomicFact"] = []
    for fact in facts:
        fields = kind_fields(fact, display_min_confidence=display_min_confidence)
        if fields["memory_kind"] == kind:
            kept.append(fact)
            if len(kept) >= limit:
                break
    return kept


def _windowed_kind_fetch(
    fetch_fn, limit: int, kind: str, *, display_min_confidence: float,
) -> tuple[list["AtomicFact"], bool]:
    """Grow the fetch window until ``limit`` matches are found, the pool is
    exhausted, or ``WINDOWED_FETCH_HARD_CAP`` is reached. Returns
    ``(facts, truncated)``.

    Neither ``get_all_facts`` nor ``search_facts_fts`` take an OFFSET — both
    are ``ORDER BY ... LIMIT ?`` from the top — so a bigger window is a
    strict superset of a smaller one. Growing the window re-fetches rather
    than paginating; that is the honest trade for "no SQL form of the full
    ``kind_fields`` precedence (confirmed / suggested-above-threshold /
    rules' no-confidence suggestion / legacy fact_type mapping) exists",
    which the WP8 LLD allows this loop to stand in for.
    """
    window = min(overfetch_limit(limit), WINDOWED_FETCH_HARD_CAP)
    if window <= 0:
        return [], False
    while True:
        batch = fetch_fn(window)
        matched = _kind_matching_facts(
            batch, kind, limit, display_min_confidence=display_min_confidence)
        if len(matched) >= limit:
            return matched, False
        if len(batch) < window:
            # Fewer rows came back than we asked for: the store has nothing
            # left to look at, so this IS the complete answer.
            return matched, False
        if window >= WINDOWED_FETCH_HARD_CAP:
            return matched, True
        window = min(window * _WINDOW_GROWTH_FACTOR, WINDOWED_FETCH_HARD_CAP)


def list_recent_facts(
    db: "DatabaseManager", profile_id: str, limit: int, kind: str | None, *,
    display_min_confidence: float = _DEFAULT_DISPLAY_MIN_CONFIDENCE,
    truncated: list[bool] | None = None,
) -> list["AtomicFact"]:
    """``get_all_facts``, newest first, with an optional displayed-kind filter.

    When ``kind`` is set, grows the fetch window (``_windowed_kind_fetch``)
    until ``limit`` matches are found or the pool is exhausted, so an older
    match is never missed just because newer non-matching memories outnumber
    the old fixed 3x/100 over-fetch window (4.1.19 L2-13 / Muse M2). When it
    is not, behaviour is byte-identical to calling ``get_all_facts`` directly.

    ``truncated``: pass a list to receive one ``bool`` — ``True`` only when
    ``WINDOWED_FETCH_HARD_CAP`` was hit before ``limit`` was filled and the
    store was not exhausted, meaning more matches may exist that this call
    did not look far enough to find. Callers that can surface this (MCP,
    CLI, HTTP) should, rather than returning a short answer with no sign it
    might be incomplete.
    """
    if not kind:
        return db.get_all_facts(profile_id, limit=limit)
    facts, was_truncated = _windowed_kind_fetch(
        lambda n: db.get_all_facts(profile_id, limit=n), limit, kind,
        display_min_confidence=display_min_confidence,
    )
    if truncated is not None:
        truncated.append(was_truncated)
    return facts


def search_facts(
    db: "DatabaseManager", query: str, profile_id: str, limit: int, kind: str | None, *,
    display_min_confidence: float = _DEFAULT_DISPLAY_MIN_CONFIDENCE,
    truncated: list[bool] | None = None,
) -> list["AtomicFact"]:
    """``search_facts_fts`` with an optional displayed-kind filter.

    Same windowed over-fetch contract (and ``truncated`` out-param) as
    ``list_recent_facts``.
    """
    if not kind:
        return db.search_facts_fts(query, profile_id, limit=limit)
    facts, was_truncated = _windowed_kind_fetch(
        lambda n: db.search_facts_fts(query, profile_id, limit=n), limit, kind,
        display_min_confidence=display_min_confidence,
    )
    if truncated is not None:
        truncated.append(was_truncated)
    return facts


def kind_item(fact: "AtomicFact") -> dict[str, Any]:
    """The five ``kind_fields`` for one fact, for merging into a result dict."""
    return kind_fields(fact)


__all__ = [
    "InvalidKind", "engine_display_min_confidence", "kind_item", "list_recent_facts",
    "resolve_kind", "search_facts",
]
