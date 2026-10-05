# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The fire-alarm path finds a release by its version number too.

``_sqlite_emergency_recall`` drops one-character words from the question, so
it used to search "4.1.20" as the single word "20" and could not tell 4.1.20
from 4.1.0, 4.1.2 or a slot at 1:20. It now matches the version whole, the
same way the recall channel does (``storage.fts_terms``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.migrations import M011_archive_and_merge
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"


@pytest.fixture()
def store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "memory.db"))
    create_all_tables(conn)
    conn.executescript(M011_archive_and_merge.DDL)
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m1', ?, 'source')", (_PROFILE,),
    )
    rows = [
        ("f-4.1.0", "4.1.0 added working memory"),
        ("f-4.1.20", "4.1.20 added answer check"),
        ("f-4.1.2", "4.1.2 fixed M043"),
        # Shorter than the release note and holds its "20": it won before.
        ("numbers-only", "Lunch at 1:20"),
        *((f"filler-{i:02d}", f"Room booked for item {i}, slot 1:20")
          for i in range(20)),
    ]
    for fid, content in rows:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " scope, created_at) VALUES "
            "(?, 'm1', ?, ?, 'personal', datetime('now'))",
            (fid, _PROFILE, content),
        )
    conn.commit()
    conn.close()
    return tmp_path


def test_a_version_question_reaches_that_release_first(store_dir: Path) -> None:
    from superlocalmemory.mcp.tools_active import _sqlite_emergency_recall

    got = _sqlite_emergency_recall("what's in 4.1.20", limit=5, profile_id=_PROFILE)
    assert got.results[0].fact.fact_id == "f-4.1.20", [
        item.fact.fact_id for item in got.results
    ]
