# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The visible-memory count is not recounted while nothing was committed.

Recall asked for it two or three times a call (Hopfield's path choice, the graph
stage), each a full ``COUNT(*)``: about 97 ms a recall on a 22k-fact store when
the warm in-memory count was not built yet. A count taken since the last commit
is exact (storage/store_signature); any commit, and any count inside a
transaction (which can see its own uncommitted rows), counts again.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


def _db(tmp_path):
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    mid = db.store_memory(MemoryRecord(profile_id="default", content="x"))
    db.store_fact(AtomicFact(profile_id="default", memory_id=mid, content="one",
                             fact_type=FactType.SEMANTIC))
    return db, mid


def _spy(db, monkeypatch) -> list[str]:
    calls: list[str] = []
    real = db.execute

    def spy(sql, *a, **k):
        if "COUNT(*)" in sql and "atomic_facts" in sql:
            calls.append(sql)
        return real(sql, *a, **k)

    monkeypatch.setattr(db, "execute", spy)
    return calls


def test_an_unchanged_store_is_counted_once(tmp_path, monkeypatch) -> None:
    db, _ = _db(tmp_path)
    calls = _spy(db, monkeypatch)
    assert [db.get_fact_count("default") for _ in range(4)] == [1, 1, 1, 1]
    assert len(calls) == 1


def test_a_commit_from_another_connection_is_counted(tmp_path, monkeypatch) -> None:
    db, mid = _db(tmp_path)
    assert db.get_fact_count("default") == 1
    other = sqlite3.connect(db.db_path)
    other.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, fact_type) "
                  "VALUES ('f-other', ?, 'default', 'two', 'semantic')", (mid,))
    other.commit()
    other.close()
    assert db.get_fact_count("default") == 2


def test_inside_a_transaction_it_always_counts(tmp_path, monkeypatch) -> None:
    db, mid = _db(tmp_path)
    assert db.get_fact_count("default") == 1
    with db.transaction():
        db.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, fact_type) "
                   "VALUES ('f-txn', ?, 'default', 'three', 'semantic')", (mid,))
        assert db.get_fact_count("default") == 2  # its own uncommitted row
