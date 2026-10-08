# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
""""Which memories are newest?" is answered from an index, not from every row.

The recency fallback of the time channel asks for the 50 newest visible facts
of a profile. With no index on ``created_at`` SQLite read every fact of the
profile, embedding and Fisher vectors included, to sort them: 5.8 s on a cold
22k-fact store, the slowest step of the first recalls after a daemon start.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from superlocalmemory.retrieval.temporal_channel import TemporalChannel
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"


def _store(tmp_path: Path) -> DatabaseManager:
    """A store as an upgrade finds it: tables exist, visibility columns arrived later."""
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.execute("ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live'")
    conn.execute("DROP INDEX IF EXISTS idx_facts_recent")
    conn.commit()
    create_all_tables(conn)  # the next engine start
    conn.commit()
    conn.close()
    return DatabaseManager(path)


def _seed(db: DatabaseManager) -> None:
    now = datetime.now(tz=timezone.utc)
    db.execute(
        "INSERT OR IGNORE INTO memories (memory_id, profile_id, content, created_at) "
        "VALUES ('m1', ?, 'x', ?)", (_PROFILE, now.isoformat()),
    )
    rows = []
    for i in range(70):
        # Pairs share a timestamp so the fact_id tie-break is exercised.
        created = (now - timedelta(hours=i // 2)).isoformat()
        archived = "archived" if i == 3 else "live"
        quarantined = 1 if i == 5 else 0
        rows.append((f"f{i:03d}", "m1", _PROFILE, f"fact {i}", created, archived, quarantined))
    for row in rows:
        db.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, created_at, "
            "archive_status, quarantined) VALUES (?, ?, ?, ?, ?, ?, ?)", row,
        )


def _recency_sql(db: DatabaseManager) -> tuple[str, tuple]:
    seen: list[tuple[str, tuple]] = []
    real = db.execute

    def spy(sql, params=()):
        seen.append((sql, tuple(params)))
        return real(sql, params)

    db.execute = spy  # type: ignore[method-assign]
    try:
        TemporalChannel(db)._recency_fallback(_PROFILE, False, False)
    finally:
        db.execute = real  # type: ignore[method-assign]
    [(sql, params)] = [(s, p) for s, p in seen if "ORDER BY af.created_at DESC" in s]
    return sql, params


def test_the_newest_memories_are_read_from_an_index(tmp_path: Path) -> None:
    db = _store(tmp_path)
    _seed(db)
    sql, params = _recency_sql(db)
    rows = db.execute("EXPLAIN QUERY PLAN " + sql, params)
    plan = " | ".join(str(dict(r)["detail"]) for r in rows)
    assert "COVERING INDEX idx_facts_recent" in plan, plan


def test_the_index_changes_no_answer(tmp_path: Path) -> None:
    db = _store(tmp_path)
    _seed(db)
    with_index = TemporalChannel(db)._recency_fallback(_PROFILE, False, False)
    db.execute("DROP INDEX idx_facts_recent")
    without_index = TemporalChannel(db)._recency_fallback(_PROFILE, False, False)
    ids = [fid for fid, _ in with_index]  # scores decay with the clock between the calls
    assert ids == [fid for fid, _ in without_index]
    assert len(ids) == 50 and "f003" not in ids and "f005" not in ids


def test_a_store_without_the_columns_still_starts(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "old.db"))
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT)")
    from superlocalmemory.storage import recency_index

    assert recency_index.ensure(conn) is False  # nothing to index, no error
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")]
    assert "idx_facts_recent" not in names
