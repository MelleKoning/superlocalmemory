# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""The three summaries read only memories a person may be shown, on their own day.

WHY THIS EXISTS (issue #113, checked on 4.1.20)
-----------------------------------------------
Session Summary, Daily Reflection and Project Work Log hand-roll their SQL
against ``atomic_facts``. 4.0.10 made every read path drop withheld rows (a
model's refusals, quarantined) and soft-deleted rows, and enumerated the paths
in tests/test_storage/test_no_read_path_shows_a_withheld_row.py. The summary
generators were not on that list, so a "Today" summary quoted a withheld
refusal as one of the owner's notes and listed its id as a source.

Daily Reflection also bucketed by the UTC calendar day while "today" was the
local day. East of UTC (India, +05:30) everything saved between local midnight
and 05:30 belonged to the previous day's reflection. The dashboard activity
chart had the same defect and was fixed in 4.1.20 with the caller's offset;
the reflection takes the same offset now.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.schema import create_all_tables
from superlocalmemory.storage.schema_v347 import apply_v347_schema
from superlocalmemory.summaries import (
    generate_daily_reflection,
    generate_project_work_log,
    generate_session_summary,
)
from superlocalmemory.summaries.sessions import list_recent_sessions

_P = "default"
_PROJECT = "/work/widget"
_REFUSAL = "Unfortunately, there is no information available about 'Gateway'."


def _schema(conn: sqlite3.Connection) -> None:
    """The live schema, plus the soft-delete column a deferred migration adds."""
    create_all_tables(conn)
    apply_v347_schema(conn)                      # tool_events
    cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
    if "archive_status" not in cols:
        conn.execute("ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT "
                     "DEFAULT 'live'")


def _store(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db))
    _schema(conn)
    conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                 "VALUES ('m1', ?, 'source')", (_P,))
    rows = [
        # fact_id, content, quarantined, archive_status, created_at (UTC)
        ("keep-1", "Shipped the widget release after both audits.", 0, "live",
         "2026-10-05T09:00:00+00:00"),
        ("keep-2", "Decided the widget cache lives in learning.db.", 0, "live",
         "2026-10-05T10:00:00+00:00"),
        ("hide-q", _REFUSAL, 1, "live", "2026-10-05T11:00:00+00:00"),
        ("hide-a", "A note the owner deleted.", 0, "archived",
         "2026-10-05T12:00:00+00:00"),
    ]
    for fid, content, quarantined, archive, created in rows:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
            "quarantined, archive_status, session_id, importance, created_at) "
            "VALUES (?, 'm1', ?, ?, ?, ?, 'sess-1', 0.5, ?)",
            (fid, _P, content, quarantined, archive, created),
        )
    conn.execute(
        "INSERT INTO tool_events (session_id, profile_id, project_path, tool_name, "
        "event_type, created_at) VALUES ('sess-1', ?, ?, 'Bash', 'call', "
        "'2026-10-05T09:00:00+00:00')", (_P, _PROJECT))
    conn.commit()
    conn.close()
    return db


def _hidden(ids: list[str]) -> list[str]:
    return [i for i in ids if i.startswith("hide-")]


class TestWithheldAndDeletedRowsAreNotSummarised:
    def test_daily_reflection(self, tmp_path) -> None:
        result = generate_daily_reflection(_store(tmp_path), "2026-10-05", _P)
        assert _hidden(result.source_fact_ids) == []
        assert sorted(result.source_fact_ids) == ["keep-1", "keep-2"]
        assert "no information available" not in result.content
        assert "deleted" not in result.content

    def test_session_summary(self, tmp_path) -> None:
        result = generate_session_summary(_store(tmp_path), "sess-1", _P)
        assert _hidden(result.source_fact_ids) == []
        assert sorted(result.source_fact_ids) == ["keep-1", "keep-2"]

    def test_project_work_log(self, tmp_path) -> None:
        result = generate_project_work_log(_store(tmp_path), _PROJECT, _P)
        assert _hidden(result.source_fact_ids) == []
        assert sorted(result.source_fact_ids) == ["keep-1", "keep-2"]


class TestDailyReflectionUsesTheCallersDay:
    def _late_evening_store(self, tmp_path: Path) -> Path:
        db = tmp_path / "memory.db"
        conn = sqlite3.connect(str(db))
        _schema(conn)
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                     "VALUES ('m', ?, 'source')", (_P,))
        for fid, created in (
            # 23:30 on 5 Oct in India, 18:00 UTC
            ("ist-5th", "2026-10-05T18:00:00+00:00"),
            # 01:30 and 02:00 on 6 Oct in India, still 5 Oct in UTC
            ("ist-6th-a", "2026-10-05T20:00:00+00:00"),
            ("ist-6th-b", "2026-10-05T20:30:00+00:00"),
        ):
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
                " importance, created_at) VALUES (?, 'm', ?, ?, 0.5, ?)",
                (fid, _P, f"note {fid}", created))
        conn.commit()
        conn.close()
        return db

    def test_an_offset_moves_the_day_boundary(self, tmp_path) -> None:
        db = self._late_evening_store(tmp_path)
        sixth = generate_daily_reflection(db, "2026-10-06", _P, tz_offset_minutes=330)
        assert sorted(sixth.source_fact_ids) == ["ist-6th-a", "ist-6th-b"]
        fifth = generate_daily_reflection(db, "2026-10-05", _P, tz_offset_minutes=330)
        assert fifth.source_fact_ids == ["ist-5th"]
        assert sixth.metadata["tz_offset_minutes"] == 330

    def test_no_offset_keeps_the_utc_day(self, tmp_path) -> None:
        db = self._late_evening_store(tmp_path)
        result = generate_daily_reflection(db, "2026-10-05", _P)
        assert sorted(result.source_fact_ids) == ["ist-5th", "ist-6th-a", "ist-6th-b"]

    @pytest.mark.parametrize("bad", [841, -841, 1.5, "330", True])
    def test_an_impossible_offset_is_refused(self, tmp_path, bad) -> None:
        with pytest.raises(ValueError, match="offset"):
            generate_daily_reflection(self._late_evening_store(tmp_path), "2026-10-06",
                                      _P, tz_offset_minutes=bad)


class TestSessionsCanBeFound:
    """A session summary needs a session id, and nothing listed them."""

    def test_lists_sessions_with_visible_memories_only(self, tmp_path) -> None:
        db = _store(tmp_path)
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
            "quarantined, session_id, created_at) VALUES ('only-hidden', 'm1', ?, "
            "?, 1, 'sess-hidden', '2026-10-06T00:00:00+00:00')", (_P, _REFUSAL))
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
            "session_id, created_at) VALUES ('other', 'm1', 'someone-else', 'x', "
            "'sess-other', '2026-10-06T00:00:00+00:00')")
        conn.commit()
        conn.close()
        sessions = list_recent_sessions(db, _P)
        assert [s["session_id"] for s in sessions] == ["sess-1"]
        assert sessions[0]["memory_count"] == 2
        assert sessions[0]["last_at"] == "2026-10-05T10:00:00+00:00"

    def test_order_is_newest_first_and_repeatable(self, tmp_path) -> None:
        db = _store(tmp_path)
        conn = sqlite3.connect(str(db))
        for sid in ("sess-b", "sess-a"):
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
                "session_id, created_at) VALUES (?, 'm1', ?, 'x', ?, "
                "'2026-10-07T00:00:00+00:00')", (f"f-{sid}", _P, sid))
        conn.commit()
        conn.close()
        first = list_recent_sessions(db, _P)
        assert [s["session_id"] for s in first] == ["sess-a", "sess-b", "sess-1"]
        assert first == list_recent_sessions(db, _P)

    def test_a_missing_store_is_an_empty_list_not_an_error(self, tmp_path) -> None:
        assert list_recent_sessions(tmp_path / "nope.db", _P) == []
