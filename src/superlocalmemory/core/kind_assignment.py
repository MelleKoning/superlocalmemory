# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Putting a memory kind on facts as they are saved and enriched.

A kind the caller declares on a save is confirmed and applies to every fact of
that memory. Otherwise the classifier may suggest one; a suggestion is a label
only and never changes ``fact_type`` (LLD invariant I2). Nothing here raises
into a write, and nothing is written to a store without the kind columns (I1,
I3).
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import UTC, datetime
from typing import Any, Sequence

from superlocalmemory.storage.memory_kinds import (
    COARSE,
    KIND_COLUMNS,
    METADATA_KEY,
    KindAssignment,
    KindSource,
    MemoryKind,
    is_confirmed,
    parse_kind,
)

logger = logging.getLogger(__name__)

#: Saves whose declared kind is trusted as confirmed. A confirmed rule reaches
#: every later session, so automatic captures (hooks, observers) and file
#: imports cannot confirm one; their facts stay open to suggestions.
KIND_TRUSTED_SOURCES = frozenset({
    "http", "python-api", "python-api-prebuilt", "mcp-offline-worker", "restore-reimport",
})


def caller_kind(metadata: dict | None, source_type: str) -> MemoryKind | None:
    """The kind declared on a trusted save, or None."""
    declared = (metadata or {}).get(METADATA_KEY)
    if declared is None or source_type not in KIND_TRUSTED_SOURCES:
        return None
    kind = parse_kind(declared)
    if kind is None:
        logger.warning("Ignored an unknown memory kind on save; the memory is stored untyped")
    return kind


def _has_columns(db: Any) -> bool:
    try:
        return bool(db.has_memory_kind_columns())
    except Exception:  # noqa: BLE001 - a probe failure must not fail the write
        return False


def with_assignment(fact: Any, assignment: KindAssignment | None, now_iso: str) -> Any:
    """``fact`` carrying ``assignment``: confirmed kinds also set ``fact_type``.

    ``assignment`` is ``None`` whenever there is nothing to apply — kinds
    turned off, no classifier wired, or a suggestion attempt that yielded
    nothing (L2-12). That must also blank any kind ``fact`` already carries,
    not merely withhold a new one: the Mode B/C extraction call can already
    have attached a ``model:llm`` hint (``encoding.llm_kind_hint.with_hint``)
    before ``assign_kinds`` ever runs, and a disabled backend must store no
    machine kind at all — the same clearing
    ``encoding.memory_kind_classifier.with_kind`` already documents for its
    own backend-off case. A caller-declared kind is never affected: that
    path always produces a concrete CALLER assignment, never ``None``
    (``_suggestions`` returns one for every fact once a kind is declared).
    """
    if assignment is None:
        return dataclasses.replace(fact, **{name: None for name in KIND_COLUMNS})
    from superlocalmemory.storage.models import FactType

    columns = assignment.as_columns(now_iso)
    if is_confirmed(assignment.source.value):
        return dataclasses.replace(fact, fact_type=FactType(COARSE[assignment.kind]), **columns)
    return dataclasses.replace(fact, **columns)


def _suggestions(facts: Sequence[Any], declared: MemoryKind | None,
                 classifier: Any) -> list[KindAssignment | None]:
    if declared is not None:
        return [KindAssignment(declared, KindSource.CALLER, None, "caller")] * len(facts)
    if classifier is None:
        return [None] * len(facts)
    try:
        out = list(classifier.suggest(facts, caller_kind=None))
    except Exception as exc:  # noqa: BLE001 - typing never fails a save
        logger.warning("Memory kind suggestion skipped (%s)", type(exc).__name__)
        return [None] * len(facts)
    if len(out) != len(facts):
        return [None] * len(facts)
    return out


def assign_kinds(facts: list, *, metadata: dict | None, source_type: str,
                 classifier: Any, db: Any) -> list:
    """``facts`` with their kinds. Unchanged when the store has no kind columns."""
    if not facts or not _has_columns(db):
        return facts
    now = datetime.now(UTC).isoformat()
    declared = caller_kind(metadata, source_type)
    return [with_assignment(fact, assignment, now)
            for fact, assignment in zip(facts, _suggestions(facts, declared, classifier))]


def kind_update_columns(fact: Any, db: Any) -> dict[str, Any]:
    """The kind columns to write with a fact update, or {} (I3)."""
    if getattr(fact, "memory_kind", None) is None or not _has_columns(db):
        return {}
    return {name: getattr(fact, name, None) for name in (
        "memory_kind", "memory_kind_source", "memory_kind_confidence",
        "memory_kind_recipe", "memory_kind_at")}


__all__ = ["KIND_TRUSTED_SOURCES", "assign_kinds", "caller_kind", "kind_update_columns",
           "with_assignment"]
