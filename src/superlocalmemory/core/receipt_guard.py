# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A saved memory never loses its own first-queryable fact to another memory.

At save time a memory gets one fact that makes it searchable at once. During
enrichment that fact is consolidated against older facts; when it was judged a
near-duplicate of an OLDER memory's fact, it used to be deleted. Notes that
differ only in a number or a declared kind ("channel 4" / "channel 9") were
judged near-duplicates, so the newer memory ended with no fact at all: found
right after saving, gone a moment later, its number and kind lost.

A near-duplicate verdict against another memory now keeps this memory's own
fact (the older fact still counts the repeat). A verdict against a fact of the
same save is unaffected: that save keeps another fact of its own.
"""

from __future__ import annotations

from collections.abc import Collection


def keeps_own_fact(action: object, is_queryable_promotion: bool,
                   created_ids: Collection[str]) -> bool:
    """True when ``action`` would delete this memory's own fact for another's."""
    kind = getattr(getattr(action, "action_type", None), "value", None)
    target = getattr(action, "existing_fact_id", "") or ""
    return (kind == "noop" and is_queryable_promotion and bool(target)
            and target not in created_ids)
