# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Q5 (2026-10-06): the "Changed This Week" insight row carries the memory
kind, not the raw legacy ``fact_type``.

Reproduced live: ``GET /api/v3/insights/changed_this_week`` (the dashboard's
quick-insight "Changed This Week" panel, rendered by
``ui/js/quick-actions.js::renderChangedThisWeek``) selected only
``fact_type`` from ``atomic_facts`` and returned it verbatim, so every row
showed "episodic"/"semantic"/"opinion"/"prospective" instead of the memory
kind ("Decision", "Rule", "Preference", ...) every other surface (recall,
the CLI, MCP, the Memories table -- see ``storage/memory_kinds.py
kind_fields()``, "THE serializer every surface uses") already shows.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.server.routes.insights import _action_changed_this_week


def _conn_with_schema(tmp_path, *, with_kind_columns: bool) -> sqlite3.Connection:
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    columns = [
        "fact_id TEXT PRIMARY KEY",
        "profile_id TEXT",
        "content TEXT",
        "fact_type TEXT",
        "created_at TEXT",
        "session_id TEXT",
        "confidence REAL",
    ]
    if with_kind_columns:
        columns += [
            "memory_kind TEXT",
            "memory_kind_source TEXT",
            "memory_kind_confidence REAL",
        ]
    conn.execute(f"CREATE TABLE atomic_facts ({', '.join(columns)})")
    conn.commit()
    return conn


def _insert(conn, **fields) -> None:
    columns = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    conn.execute(
        f"INSERT INTO atomic_facts ({columns}) VALUES ({placeholders})",
        list(fields.values()),
    )
    conn.commit()


class TestChangedThisWeekShowsMemoryKind:
    def test_confirmed_kind_replaces_the_raw_fact_type(self, tmp_path) -> None:
        conn = _conn_with_schema(tmp_path, with_kind_columns=True)
        _insert(
            conn,
            fact_id="f1", profile_id="default", content="We ship Tuesdays.",
            fact_type="episodic", created_at="2026-10-06T00:00:00Z",
            session_id="s1", confidence=0.9,
            memory_kind="decision", memory_kind_source="user",
            memory_kind_confidence=None,
        )
        try:
            result = _action_changed_this_week(conn, "default", limit=50, days=7)
        finally:
            conn.close()

        assert result["count"] == 1
        item = result["items"][0]
        assert item["fact_type"] == "episodic", "the legacy field stays available"
        assert item["memory_kind_label"] == "Decision", (
            f"expected the Decision label, got {item.get('memory_kind_label')!r}"
        )
        assert item["memory_kind_state"] == "confirmed"

    def test_untyped_legacy_row_falls_back_through_kind_fields_not_raw_fact_type(
        self, tmp_path,
    ) -> None:
        """A row with no memory_kind at all still gets a label, resolved by
        kind_fields()'s own legacy-fact_type mapping -- not by the route
        just echoing the bare column."""
        conn = _conn_with_schema(tmp_path, with_kind_columns=True)
        _insert(
            conn,
            fact_id="f2", profile_id="default", content="Met Bob for coffee.",
            fact_type="episodic", created_at="2026-10-06T00:00:00Z",
            session_id="s1", confidence=0.5,
            memory_kind=None, memory_kind_source=None, memory_kind_confidence=None,
        )
        try:
            result = _action_changed_this_week(conn, "default", limit=50, days=7)
        finally:
            conn.close()

        item = result["items"][0]
        assert "memory_kind_label" in item
        assert item["memory_kind_state"] == "legacy"

    def test_pre_419_store_with_no_kind_columns_does_not_error(self, tmp_path) -> None:
        """A store from before the M052 kind migration has no memory_kind
        column at all; the query must degrade to fact_type only, never
        raise on an unknown column (which would silently empty the whole
        panel -- a worse regression than showing the legacy label)."""
        conn = _conn_with_schema(tmp_path, with_kind_columns=False)
        _insert(
            conn,
            fact_id="f3", profile_id="default", content="Old store fact.",
            fact_type="semantic", created_at="2026-10-06T00:00:00Z",
            session_id="s1", confidence=0.7,
        )
        try:
            result = _action_changed_this_week(conn, "default", limit=50, days=7)
        finally:
            conn.close()

        assert result["count"] == 1
        item = result["items"][0]
        assert item["fact_type"] == "semantic"
        assert "memory_kind_label" not in item
