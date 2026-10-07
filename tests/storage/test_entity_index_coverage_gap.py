# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Q7 (2026-10-06): the entity-index downgrade/restore gap, and its repair.

Reproduced on a synthetic store: the one-time coverage backfill
(``storage/entity_index.py::backfill``, keyed ``REPAIR_KEY``) snapshots
``MAX(rowid)`` the first time it runs and never revisits that snapshot once
it marks itself ``complete``. A fact written AFTER that snapshot by any path
that bypasses the real-time ``record_fact_entities`` hook -- an older
version reached by a downgrade, a restore, or a direct import -- is
permanently invisible to it: ``backfill()`` keeps reporting ``complete`` and
does nothing for that fact, forever.

``repair_coverage_gap`` fixes this with a SEPARATE, never-terminal sweep
(``GAP_REPAIR_KEY``) that re-derives its target to the current
``MAX(rowid)`` at the end of every lap and wraps its cursor back to the
start, so it keeps re-checking the whole table indefinitely and will catch
a gap introduced at any time.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import entity_index

_SCHEMA = """
CREATE TABLE atomic_facts (
    fact_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL DEFAULT 'default',
    canonical_entities_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE canonical_entities (
    entity_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL DEFAULT 'default',
    canonical_name TEXT NOT NULL
);
CREATE TABLE fact_entity_associations (
    profile_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    first_operation_id TEXT NOT NULL DEFAULT '',
    count_applied INTEGER NOT NULL DEFAULT 0 CHECK (count_applied IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (profile_id, fact_id, entity_id),
    FOREIGN KEY (fact_id) REFERENCES atomic_facts(fact_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES canonical_entities(entity_id) ON DELETE CASCADE
);
CREATE TABLE fact_entity_association_repair_state (
    repair_key TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'running', 'retrying', 'complete')),
    target_fact_rowid INTEGER NOT NULL DEFAULT -1,
    last_fact_rowid INTEGER NOT NULL DEFAULT 0,
    scanned INTEGER NOT NULL DEFAULT 0,
    inserted INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
"""


def _store(tmp_path: Path) -> Path:
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


def _add_entity(db_path: Path, entity_id: str, profile_id: str = "default") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO canonical_entities (entity_id, profile_id, canonical_name) "
        "VALUES (?, ?, ?)",
        (entity_id, profile_id, entity_id),
    )
    conn.commit()
    conn.close()


def _add_fact(db_path: Path, fact_id: str, entities: list[str],
             profile_id: str = "default") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO atomic_facts (fact_id, profile_id, canonical_entities_json) "
        "VALUES (?, ?, ?)",
        (fact_id, profile_id, json.dumps(entities)),
    )
    conn.commit()
    conn.close()


def _pairs(db_path: Path) -> set[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            (row[0], row[1]) for row in
            conn.execute("SELECT fact_id, entity_id FROM fact_entity_associations")
        }
    finally:
        conn.close()


class TestDowngradeGapReproduced:
    def test_backfill_alone_never_covers_a_fact_written_after_it_completed(
        self, tmp_path,
    ) -> None:
        db_path = _store(tmp_path)
        _add_entity(db_path, "kestrel")
        _add_fact(db_path, "fact-old", ["kestrel"])

        # The one-time backfill runs to completion against the facts that
        # exist right now.
        result = entity_index.backfill(db_path, batch_size=10, max_batches=5)
        assert result["complete"] is True
        assert ("fact-old", "kestrel") in _pairs(db_path)

        # A fact written afterwards by a path that bypasses the real-time
        # hook (simulating an older-version downgrade write, or a direct
        # restore/import) -- never indexed.
        _add_fact(db_path, "fact-gap", ["kestrel"])
        assert ("fact-gap", "kestrel") not in _pairs(db_path)

        # The bug: backfill() still reports complete and does NOTHING for
        # the gap fact, no matter how many times it is called.
        again = entity_index.backfill(db_path, batch_size=10, max_batches=5)
        assert again["complete"] is True
        assert again["inserted"] == 0
        assert ("fact-gap", "kestrel") not in _pairs(db_path), (
            "backfill() must not be able to see this fact -- that is the "
            "gap this test characterizes, repaired by repair_coverage_gap"
        )

    def test_repair_coverage_gap_finds_and_fills_it(self, tmp_path) -> None:
        db_path = _store(tmp_path)
        _add_entity(db_path, "kestrel")
        _add_fact(db_path, "fact-old", ["kestrel"])
        entity_index.backfill(db_path, batch_size=10, max_batches=5)
        _add_fact(db_path, "fact-gap", ["kestrel"])
        assert ("fact-gap", "kestrel") not in _pairs(db_path)

        result = entity_index.repair_coverage_gap(db_path, batch_size=10, max_batches=5)

        assert ("fact-gap", "kestrel") in _pairs(db_path)
        assert result["inserted"] >= 1

    def test_sweep_is_bounded_and_resumable_across_many_small_batches(
        self, tmp_path,
    ) -> None:
        """A single call only ever touches ``batch_size`` facts; repeated
        small calls still make full progress via the durable cursor."""
        db_path = _store(tmp_path)
        _add_entity(db_path, "harbor")
        for i in range(37):
            _add_fact(db_path, f"fact-{i}", ["harbor"])
        # None of these are indexed -- as if every one of them was written
        # by a path that bypasses the real-time hook.
        assert _pairs(db_path) == set()

        total_inserted = 0
        for _ in range(20):  # batch_size=5 needs 8 batches for 37 facts + 1 lap-end
            result = entity_index.repair_coverage_gap(db_path, batch_size=5, max_batches=1)
            total_inserted += result["inserted"]
            if result["scanned"] == 0 and result["laps_completed"]:
                break

        pairs = _pairs(db_path)
        assert len(pairs) == 37, f"expected all 37 facts indexed, got {len(pairs)}"
        assert total_inserted == 37

    def test_sweep_wraps_around_and_catches_a_gap_introduced_after_a_lap(
        self, tmp_path,
    ) -> None:
        """The sweep must not stop for good after its first full lap: a gap
        introduced AFTER that lap (e.g. a later downgrade write) must still
        be caught on a subsequent lap."""
        db_path = _store(tmp_path)
        _add_entity(db_path, "tundra")
        _add_fact(db_path, "fact-a", ["tundra"])

        # First lap: covers fact-a, then reports lap_complete.
        first = entity_index.repair_coverage_gap(db_path, batch_size=10, max_batches=5)
        assert ("fact-a", "tundra") in _pairs(db_path)
        assert first["laps_completed"] >= 1

        # A new gap fact appears after the lap finished.
        _add_fact(db_path, "fact-b", ["tundra"])
        assert ("fact-b", "tundra") not in _pairs(db_path)

        # The NEXT call(s) must pick it up -- the sweep re-derives its
        # target and wraps, rather than staying frozen at the old one.
        second = entity_index.repair_coverage_gap(db_path, batch_size=10, max_batches=5)
        assert ("fact-b", "tundra") in _pairs(db_path), (
            f"gap fact not covered after wrap-around: {second}"
        )

    def test_already_indexed_facts_cost_no_duplicate_inserts(self, tmp_path) -> None:
        """Re-sweeping fully-covered facts must be a cheap no-op, not an
        error or a duplicate row (the primary key would refuse a literal
        duplicate; INSERT OR IGNORE is what keeps this safe to repeat)."""
        db_path = _store(tmp_path)
        _add_entity(db_path, "juniper")
        _add_fact(db_path, "fact-x", ["juniper"])
        entity_index.backfill(db_path, batch_size=10, max_batches=5)
        assert ("fact-x", "juniper") in _pairs(db_path)

        result = entity_index.repair_coverage_gap(db_path, batch_size=10, max_batches=3)
        assert result["inserted"] == 0
        assert _pairs(db_path) == {("fact-x", "juniper")}
