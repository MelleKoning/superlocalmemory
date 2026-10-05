# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Project summaries must be reachable and must not overstate their coverage.

WHY THIS EXISTS
---------------
Two defects shipped together in 4.0.8's Summaries tab.

**Unreachable.** The pane offered a "This project" button that sent an empty
target, and the endpoint answers an empty project target with HTTP 400
"project requires target". It could never have worked: SLM runs as one global
daemon and the dashboard is a browser tab, so there is no working directory for
"this" to mean. The fix is a picker fed by projects the store has actually
observed, which is what ``/api/summary/projects`` exists to serve.

**Overstated.** Coverage was ``FULL`` when *either* tool events or facts were
present. A project with 86 tool events and no stored facts therefore rendered as
"Built from 0 memories · coverage: full" — a claim of complete coverage over
nothing. Project logs describe two things, what was done and what was learned;
"full" now requires both.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.summaries import generate_project_work_log
from superlocalmemory.summaries.base import (
    COVERAGE_FULL,
    COVERAGE_INSUFFICIENT,
    COVERAGE_PARTIAL,
)

PROJECT = "/Users/someone/code/widget"


def _make_db(tmp_path, *, events: int, facts: int, profile: str = "default"):
    """Minimal memory.db carrying exactly the columns the generator reads.

    Facts are joined to tool events by ``session_id``, so a fact only counts for
    a project when a tool event in that project shares its session. Facts are
    therefore attached to session ``s0``, which means ``events=0, facts>0``
    yields no visible facts — a property of the real query, not a fixture quirk.
    """
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE tool_events (
            id INTEGER PRIMARY KEY, session_id TEXT, profile_id TEXT,
            project_path TEXT, tool_name TEXT, event_type TEXT,
            input_summary TEXT, output_summary TEXT, duration_ms REAL,
            metadata TEXT, created_at TEXT
        );
        CREATE TABLE atomic_facts (
            fact_id TEXT PRIMARY KEY, memory_id TEXT, profile_id TEXT,
            session_id TEXT, content TEXT, fact_type TEXT, confidence REAL,
            importance REAL, lifecycle TEXT, canonical_entities_json TEXT,
            created_at TEXT
        );
        CREATE TABLE memories (
            memory_id TEXT PRIMARY KEY, profile_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    for i in range(events):
        conn.execute(
            "INSERT INTO tool_events (session_id, profile_id, project_path,"
            " tool_name, event_type, created_at) VALUES (?,?,?,?,?,?)",
            (f"s{i}", profile, PROJECT, "Bash", "call", "2026-08-17T10:00:00Z"),
        )
    for i in range(facts):
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, session_id,"
            " content, fact_type, confidence, importance, lifecycle,"
            " canonical_entities_json, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f"f{i}", f"m{i}", profile, "s0", f"Widget fact {i}", "semantic",
             0.9, 0.8, "warm", "[]", "2026-08-17T10:00:00Z"),
        )
    conn.commit()
    conn.close()
    return db


class TestCoverageHonesty:
    def test_activity_without_facts_is_partial_not_full(self, tmp_path):
        """The exact shape that rendered as '0 memories · coverage: full'."""
        db = _make_db(tmp_path, events=86, facts=0)
        result = generate_project_work_log(db, PROJECT)
        assert result.coverage == COVERAGE_PARTIAL
        assert result.source_fact_ids == []

    def test_facts_are_invisible_without_a_matching_tool_event(self, tmp_path):
        """Project scope flows through tool_events.project_path, so facts alone
        are not reachable. Documented here so the asymmetry is not mistaken for
        a bug later."""
        db = _make_db(tmp_path, events=0, facts=5)
        result = generate_project_work_log(db, PROJECT)
        assert result.coverage == COVERAGE_INSUFFICIENT
        assert result.source_fact_ids == []

    def test_both_present_is_full(self, tmp_path):
        db = _make_db(tmp_path, events=12, facts=5)
        result = generate_project_work_log(db, PROJECT)
        assert result.coverage == COVERAGE_FULL

    def test_neither_present_is_insufficient(self, tmp_path):
        db = _make_db(tmp_path, events=0, facts=0)
        result = generate_project_work_log(db, PROJECT)
        assert result.coverage == COVERAGE_INSUFFICIENT

    @pytest.mark.parametrize(
        "events,facts", [(86, 0), (0, 5), (12, 5), (0, 0), (1, 1)]
    )
    def test_coverage_full_always_implies_sources(self, tmp_path, events, facts):
        """The invariant the old rule broke: full coverage over zero sources."""
        case = tmp_path / f"e{events}f{facts}"
        case.mkdir()
        result = generate_project_work_log(
            _make_db(case, events=events, facts=facts), PROJECT
        )
        if result.coverage == COVERAGE_FULL:
            assert result.source_fact_ids, "claimed full coverage with no sources"


class TestProjectPickerEndpoint:
    """The picker replaces a control that could not work. Keep it wired."""

    def test_endpoint_is_registered(self):
        from superlocalmemory.server.routes import memories

        paths = {r.path for r in memories.router.routes}
        assert "/api/summary/projects" in paths

    def test_label_shortens_long_paths_but_keeps_siblings_distinct(self):
        from superlocalmemory.server.routes.memories import _project_label

        a = "/Users/alice/Documents/official/Call-off-tool/goep-serviceTool"
        b = "/Users/alice/Documents/official/testing - automation"
        assert _project_label(a) == "Call-off-tool/goep-serviceTool"
        assert _project_label(b) == "official/testing - automation"
        assert _project_label(a) != _project_label(b)

    @pytest.mark.parametrize("path", ["", "/", "widget"])
    def test_label_never_raises_on_degenerate_paths(self, path):
        from superlocalmemory.server.routes.memories import _project_label

        assert isinstance(_project_label(path), str)

    def test_empty_project_target_is_still_rejected(self):
        """The picker sends a real path; an empty one must stay a 400, not a
        silent summary of everything."""
        import inspect

        from superlocalmemory.server.routes import memories

        src = inspect.getsource(memories.get_summary)
        assert "project requires target" in src
