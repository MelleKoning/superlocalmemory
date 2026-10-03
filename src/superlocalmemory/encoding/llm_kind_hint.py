# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Mode B/C: ask the extraction call that already runs for a kind, too (LLD §3.6).

No extra model call. The extraction prompt gains one instruction line and one
field in its example; each extracted item's ``kind`` (if it is one of the nine,
or an accepted alias) rides on the fact as a ``model:llm`` *suggestion*. The
4-way ``fact_type`` parse is untouched — a suggestion never changes it.

The ``memory_kinds.llm_extract_kinds`` flag (default on) sends exactly the
4.1.18 prompt when off, so extraction quality can be compared with and without
the extra line (risk R18) and the line dropped if it costs anything.

Kept apart from ``fact_extractor.py`` so that file only gains a few lines.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.encoding.memory_kind_recipe import LLM_RECIPE
from superlocalmemory.storage.memory_kinds import (
    KindAssignment,
    KindSource,
    MemoryKind,
    parse_kind,
)
from superlocalmemory.storage.models import AtomicFact

_KIND_CHOICES = "|".join(kind.value for kind in MemoryKind)
KIND_INSTRUCTION = (
    f'Also give each fact a "kind": one of {_KIND_CHOICES} '
    "(rule = a standing instruction; decision = a choice that was made; procedure = "
    "steps; status = current state of ongoing work; correction = says an earlier "
    "statement was wrong).\n"
)

_RESPOND = "Respond ONLY with a JSON array."


def system_prompt(base: str, extract_kinds: bool) -> str:
    """``base`` unchanged when the flag is off; otherwise with the kind line and field.

    If ``base`` ever stops containing the anchors below, the base prompt is
    returned as is — a missing hint costs a suggestion, never an extraction.
    """
    if extract_kinds is not True or _RESPOND not in base:
        return base
    with_kinds = base.replace(_RESPOND, KIND_INSTRUCTION + _RESPOND, 1)
    for fact_type, kind in (("semantic", MemoryKind.SEMANTIC), ("opinion", MemoryKind.OPINION)):
        with_kinds = with_kinds.replace(f'"fact_type":"{fact_type}",',
                                        f'"fact_type":"{fact_type}","kind":"{kind.value}",', 1)
    return with_kinds


def hint_from_item(item: Any) -> KindAssignment | None:
    """The suggestion an extracted item carries, or None for a missing or unknown kind."""
    if not isinstance(item, dict):
        return None
    kind = parse_kind(item.get("kind"))
    if kind is None:
        return None
    return KindAssignment(kind, KindSource.MODEL_LLM, None, LLM_RECIPE)


def with_hint(fact: AtomicFact, item: Any) -> AtomicFact:
    """``fact`` carrying the item's kind suggestion in its kind fields, if it has one."""
    hint = hint_from_item(item)
    if hint is None:
        return fact
    return dataclasses.replace(fact, **hint.as_columns(datetime.now(UTC).isoformat()))


__all__ = ["KIND_INSTRUCTION", "hint_from_item", "system_prompt", "with_hint"]
