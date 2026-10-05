# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm summary``: memory ids in plain text, this computer's day, the session list."""

from __future__ import annotations

import json
import sqlite3
from argparse import Namespace

import pytest

from superlocalmemory.cli import summary_cmd
from superlocalmemory.storage.schema import create_all_tables


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(summary_cmd, "_load_config", lambda: None)
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        create_all_tables(conn)
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                     "VALUES ('m', 'default', 's')")
        for i in range(25):
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
                "session_id, importance, created_at) VALUES (?, 'm', 'default', ?, "
                "'sess-1', 0.5, '2026-10-05T20:00:00+00:00')",
                (f"fact-{i:02d}", f"note {i}"))
    return tmp_path


def _args(**kw) -> Namespace:
    return Namespace(**{"json": False, "profile": "default", **kw})


def test_plain_text_names_the_memories(store, capsys) -> None:
    summary_cmd.cmd_summary(_args(summary_command="session", session_id="sess-1"))
    out = capsys.readouterr().out
    assert "Memory ids:" in out and "fact-00" in out
    assert "… and 5 more (--json lists every one)" in out


def test_the_day_is_this_computers_day(store, capsys, monkeypatch) -> None:
    from superlocalmemory.summaries import base

    monkeypatch.setattr(base, "local_offset_minutes", lambda day=None: 330)
    summary_cmd.cmd_summary(_args(summary_command="day", date="2026-10-06", json=True))
    data = json.loads(capsys.readouterr().out)
    assert len(data["source_fact_ids"]) == 25
    assert data["metadata"]["tz_offset_minutes"] == 330


def test_an_unreadable_date_is_refused(store, capsys) -> None:
    with pytest.raises(SystemExit) as info:
        summary_cmd.cmd_summary(_args(summary_command="day", date="5th Oct"))
    assert info.value.code == 2


def test_sessions_lists_what_can_be_summarised(store, capsys) -> None:
    summary_cmd.cmd_summary(_args(summary_command="sessions"))
    out = capsys.readouterr().out
    assert "sess-1  25 memories" in out and "slm summary session <id>" in out
    summary_cmd.cmd_summary(_args(summary_command="sessions", json=True))
    assert json.loads(capsys.readouterr().out)["sessions"][0]["session_id"] == "sess-1"


def test_local_offset_is_a_whole_number_of_minutes() -> None:
    from superlocalmemory.summaries.base import local_offset_minutes

    value = local_offset_minutes("2026-10-05")
    assert isinstance(value, int) and -840 <= value <= 840
