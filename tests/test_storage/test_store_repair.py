# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The one-time repair restores blanked memories and removes only empty entities."""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.storage.store_repair import (
    apply_repair,
    is_blanked_copy,
    plan_repair,
)

ORIGINAL = "Backups go to Agentic_official/slm-backups/pre-4.1.18-<ts> with sha256."
BLANKED = "Backups go to [REDACTED:ENTROPY:.18-]<ts> with sha256."
MISMATCH = "queryable ingestion memory content mismatch"


@pytest.fixture()
def conn(tmp_path) -> sqlite3.Connection:
    c = sqlite3.connect(tmp_path / "m.db", isolation_level=None)
    c.executescript("""
        CREATE TABLE memories(memory_id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE atomic_facts(fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,
                                  canonical_entities_json TEXT);
        CREATE TABLE ingestion_operations(operation_id TEXT PRIMARY KEY, raw_content TEXT,
            queryable_fact_ids_json TEXT, state TEXT, last_error TEXT, attempt_count INT,
            next_retry_at REAL, lease_owner TEXT, lease_expires_at REAL, updated_at TEXT);
        CREATE TABLE canonical_entities(entity_id TEXT PRIMARY KEY, canonical_name TEXT,
                                        fact_count INT);
        CREATE TABLE entity_aliases(entity_id TEXT, alias TEXT, source TEXT);
    """)
    c.execute("INSERT INTO memories VALUES ('m1', ?)", (BLANKED,))
    c.execute("INSERT INTO atomic_facts VALUES ('f1', 'm1', ?, '[\"e_used\"]')", (BLANKED,))
    c.execute("INSERT INTO ingestion_operations VALUES ('op1', ?, ?, 'failed', ?, 9, 1e12, "
              "'', 0, '')", (ORIGINAL, json.dumps(["f1"]), MISMATCH))
    # A different stored text: not a blanked copy of its original -> left alone.
    c.execute("INSERT INTO memories VALUES ('m2', 'something else entirely')")
    c.execute("INSERT INTO atomic_facts VALUES ('f2', 'm2', 'something else entirely', '[]')")
    c.execute("INSERT INTO ingestion_operations VALUES ('op2', 'the real text', ?, 'failed', "
              "?, 9, 1e12, '', 0, '')", (json.dumps(["f2"]), MISMATCH))
    for eid, name, n in (("e_zorblax", "Zorblax", 0), ("e_used", "Used", 0),
                         ("e_alias", "Aliased", 0), ("e_real", "Real", 3)):
        c.execute("INSERT INTO canonical_entities VALUES (?, ?, ?)", (eid, name, n))
    c.execute("INSERT INTO entity_aliases VALUES ('e_alias', 'Al', 'llm')")
    # Every entity carries its own name as a 'canonical' self-alias.
    for eid, name in (("e_zorblax", "Zorblax"), ("e_alias", "Aliased"), ("e_used", "Used")):
        c.execute("INSERT INTO entity_aliases VALUES (?, ?, 'canonical')", (eid, name))
    return c


def test_a_blanked_copy_is_recognised_and_nothing_else_is() -> None:
    assert is_blanked_copy(BLANKED, ORIGINAL)
    assert not is_blanked_copy(ORIGINAL, ORIGINAL)            # nothing blanked
    assert not is_blanked_copy("totally different", ORIGINAL)  # no marker
    assert not is_blanked_copy("Backups go to [REDACTED:X:1] with md5.", ORIGINAL)


def test_the_plan_reads_only(conn) -> None:
    before = conn.execute("SELECT * FROM memories ORDER BY 1").fetchall()
    plan = plan_repair(conn)
    assert plan.summary() == {"memories_to_restore": 1, "memories_left_alone": 1,
                              "empty_entities_to_remove": 1}
    assert plan.empty_entities == ("e_zorblax",)
    assert conn.execute("SELECT * FROM memories ORDER BY 1").fetchall() == before


def test_apply_restores_the_memory_and_requeues_its_enrichment(conn) -> None:
    result = apply_repair(conn, plan_repair(conn))
    assert (result.memories_restored, result.memories_skipped, result.entities_removed) == (1, 0, 1)
    assert conn.execute("SELECT content FROM memories WHERE memory_id='m1'").fetchone()[0] == ORIGINAL
    assert conn.execute("SELECT content FROM atomic_facts WHERE fact_id='f1'").fetchone()[0] == ORIGINAL
    op = conn.execute("SELECT state, attempt_count, last_error, next_retry_at "
                      "FROM ingestion_operations WHERE operation_id='op1'").fetchone()
    assert op == ("queryable", 0, "", 0)
    # The one that was not a blanked copy is untouched.
    assert conn.execute("SELECT state FROM ingestion_operations WHERE operation_id='op2'"
                        ).fetchone()[0] == "failed"
    remaining = {r[0] for r in conn.execute("SELECT entity_id FROM canonical_entities")}
    assert remaining == {"e_used", "e_alias", "e_real"}
    assert conn.execute("SELECT COUNT(*) FROM entity_aliases WHERE entity_id='e_zorblax'"
                        ).fetchone()[0] == 0


def test_a_memory_edited_after_planning_is_skipped_not_overwritten(conn) -> None:
    plan = plan_repair(conn)
    conn.execute("UPDATE memories SET content='edited by the user' WHERE memory_id='m1'")
    result = apply_repair(conn, plan)
    assert (result.memories_restored, result.memories_skipped) == (0, 1)
    assert conn.execute("SELECT content FROM memories WHERE memory_id='m1'"
                        ).fetchone()[0] == "edited by the user"
