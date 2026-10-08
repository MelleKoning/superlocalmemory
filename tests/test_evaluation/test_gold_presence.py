# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A labelled answer that is not in the measured store is "answer not stored",
never a retrieval miss; the measurement says which, and when the answer was
written."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from superlocalmemory.evaluation.answer_quality import GoldQuestion
from superlocalmemory.evaluation.gold_presence import (
    NOT_STORED,
    PARTIALLY_STORED,
    STORED,
    check_presence,
    presence_summary,
)


def make_store(path: Path) -> Path:
    db = path / "memory.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE memories (memory_id TEXT PRIMARY KEY, profile_id TEXT,
                               content TEXT, created_at TEXT);
        CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, memory_id TEXT,
                                   profile_id TEXT, content TEXT, created_at TEXT);
    """)
    conn.executemany("INSERT INTO memories VALUES (?,?,?,?)", [
        ("m1", "default", "PRIVATE-TEXT", "2026-08-01 10:00:00"),
        ("m2", "default", "PRIVATE-TEXT", "2026-09-01 10:00:00"),
        ("m9", "other", "PRIVATE-TEXT", "2026-09-02 10:00:00"),
    ])
    conn.executemany("INSERT INTO atomic_facts VALUES (?,?,?,?,?)", [
        ("f1", "m1", "default", "PRIVATE-TEXT", "2026-08-01 10:00:05"),
    ])
    conn.commit()
    conn.close()
    return db


def q(qid: str, memories=(), facts=(), answerable=True) -> GoldQuestion:
    return GoldQuestion(qid=qid, question=f"question {qid}", category="c",
                        answerable=answerable, gold_memory_ids=frozenset(memories),
                        gold_fact_ids=frozenset(facts))


class TestCheckPresence:
    def test_each_status_and_its_dates(self, tmp_path) -> None:
        db = make_store(tmp_path)
        result = check_presence(db, [
            q("all", memories=["m1", "m2"]),
            q("some", memories=["m2", "gone"]),
            q("none", memories=["gone"], facts=["gone-fact"]),
            q("fact", facts=["f1"]),
            q("unanswerable", answerable=False),
        ], profile_id="default")
        assert result["all"].status == STORED
        assert result["all"].earliest_created_at == "2026-08-01 10:00:00"
        assert result["all"].latest_created_at == "2026-09-01 10:00:00"
        assert result["some"].status == PARTIALLY_STORED
        assert result["some"].missing_ids == ("gone",)
        assert result["none"].status == NOT_STORED
        assert result["none"].earliest_created_at is None
        assert result["fact"].status == STORED
        assert "unanswerable" not in result

    def test_an_id_in_another_profile_is_not_stored_here(self, tmp_path) -> None:
        db = make_store(tmp_path)
        result = check_presence(db, [q("elsewhere", memories=["m9"])], profile_id="default")
        assert result["elsewhere"].status == NOT_STORED

    def test_the_store_is_opened_read_only(self, tmp_path) -> None:
        db = make_store(tmp_path)
        db.chmod(0o444)
        try:
            check_presence(db, [q("all", memories=["m1"])], profile_id="default")
        finally:
            db.chmod(0o644)

    def test_rows_carry_no_memory_text(self, tmp_path) -> None:
        db = make_store(tmp_path)
        row = check_presence(db, [q("all", memories=["m1"])], profile_id="default")["all"].as_row()
        assert "PRIVATE-TEXT" not in json.dumps(row)
        assert set(row) == {"status", "present", "missing_ids",
                            "earliest_created_at", "latest_created_at"}


class TestSummary:
    def test_misses_are_split_by_storage(self, tmp_path) -> None:
        db = make_store(tmp_path)
        presence = check_presence(db, [
            q("found", memories=["m1"]),
            q("missed", memories=["m2"]),
            q("absent", memories=["gone"]),
        ], profile_id="default")
        summary = presence_summary(presence, {"found": 1, "missed": None, "absent": None})
        assert summary["checked"] == 3
        assert summary[STORED] == 2 and summary[NOT_STORED] == 1
        assert summary["answer_not_stored"] == ["absent"]
        assert summary["retrieval_misses"] == ["missed"]
        assert summary["misses_explained_by_storage"] == ["absent"]
