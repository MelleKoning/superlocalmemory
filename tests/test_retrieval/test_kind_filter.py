# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The shared post-retrieval kind filter (LLD/WP8 4.1.19).

recall, search and list_recent all take a ``kind`` filter that narrows on the
DISPLAYED kind (``kind_fields()['memory_kind']``), applied after enough
candidates have been over-fetched that filtering still leaves room for up to
``limit`` matches. This module is the one place that over-fetch arithmetic and
that filter loop live, so every surface agrees.
"""

from __future__ import annotations

from superlocalmemory.retrieval.kind_filter import (
    OVERFETCH_CAP,
    OVERFETCH_FACTOR,
    filter_items_by_kind,
    overfetch_limit,
)


def test_overfetch_multiplies_by_three() -> None:
    assert overfetch_limit(10) == 30
    assert overfetch_limit(1) == 3


def test_overfetch_is_capped_at_100() -> None:
    assert overfetch_limit(1000) == OVERFETCH_CAP
    assert overfetch_limit(34) == min(34 * OVERFETCH_FACTOR, OVERFETCH_CAP)


def test_overfetch_of_zero_or_negative_is_zero() -> None:
    assert overfetch_limit(0) == 0
    assert overfetch_limit(-5) == 0


def test_no_kind_passes_through_capped_at_limit() -> None:
    items = [{"memory_kind": "semantic"}, {"memory_kind": "decision"}, {"memory_kind": None}]
    assert filter_items_by_kind(items, None, 2) == items[:2]
    assert filter_items_by_kind(items, "", 10) == items


def test_kind_filter_keeps_only_matching_items_in_order() -> None:
    items = [
        {"id": 1, "memory_kind": "decision"},
        {"id": 2, "memory_kind": "semantic"},
        {"id": 3, "memory_kind": "decision"},
        {"id": 4, "memory_kind": "status"},
    ]
    assert filter_items_by_kind(items, "decision", 10) == [items[0], items[2]]


def test_kind_filter_stops_at_limit_even_with_more_matches_available() -> None:
    items = [{"id": i, "memory_kind": "rule"} for i in range(10)]
    kept = filter_items_by_kind(items, "rule", 3)
    assert kept == items[:3]


def test_kind_filter_honours_a_custom_key() -> None:
    items = [{"id": 1, "mk": "rule"}, {"id": 2, "mk": "decision"}]
    assert filter_items_by_kind(items, "rule", 10, kind_key="mk") == [items[0]]


def test_kind_filter_on_legacy_rows_uses_the_mapped_kind() -> None:
    # A legacy row has no memory_kind of its own; kind_fields() maps it from
    # fact_type before this filter ever runs, so the item it is handed already
    # carries the mapped value - this filter never re-derives anything.
    items = [{"id": 1, "memory_kind": "episodic"}]  # mapped from fact_type="episodic"
    assert filter_items_by_kind(items, "episodic", 10) == items
    assert filter_items_by_kind(items, "semantic", 10) == []
