# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair`` blanks the previews erased memories left in activity events.

Before 4.1.22 an erasure did not touch the activity log: a ``memory.stored``
event kept the first characters of the memory it announced, and a
``memory.updated`` event kept the text of the new version. The integrity scan
counted those previews but the repair left them. Damage is planted here the
way those builds left it: after the erasure, straight into the event table.

Undo contract (CHANGELOG 4.1.22: a repair never brings back anything that was
erased): the scrub keeps NO undo copy, because a copy would be the erased
words. Undoing the run therefore restores nothing for this step, and the
previews stay blank. The receipt records counts and ids only.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

_TEXT = ("Marrowick sells synthetic pears. Marrowick also repairs clocks on Fridays "
         "in the old synthetic arcade.")


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _store(engine, text: str):
    from superlocalmemory.core.engine_ingestion import canonical_store

    return canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                           require_complete=True, return_receipt=True)


_EVENTS_DDL = (  # the event bus creates it on first use (infra/event_bus.py)
    "CREATE TABLE IF NOT EXISTS memory_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "profile_id TEXT NOT NULL DEFAULT 'default', event_type TEXT NOT NULL, memory_id INTEGER, "
    "source_agent TEXT, source_protocol TEXT, payload TEXT, importance INTEGER, tier TEXT, "
    "created_at TIMESTAMP)")


def _event(conn: sqlite3.Connection, event_type: str, payload: dict) -> int:
    conn.execute(_EVENTS_DDL)
    cur = conn.execute(
        "INSERT INTO memory_events (profile_id, event_type, source_agent, source_protocol, "
        "payload, importance, tier, created_at) VALUES ('default', ?, 'materializer', "
        "'internal', ?, 5, 'hot', 't')", (event_type, json.dumps(payload)))
    return int(cur.lastrowid)


@pytest.fixture
def old_events(engine_with_mock_deps):
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    partial = _store(engine, _TEXT)
    partial_facts = list(partial.final_fact_ids)
    if len(partial_facts) < 2:
        pytest.skip("extraction produced one fact; a partial erasure needs two")
    whole = _store(engine, "Zorblax guards the lonely synthetic bridge at night.")
    live = _store(engine, "Quillon waters the synthetic fern every Tuesday.")
    successor = list(_store(engine, "Brennick now keeps the synthetic key in drawer five.")
                     .final_fact_ids)[0]
    erased_in_part = partial_facts[0]
    erased_whole = list(whole.final_fact_ids)
    for fact_id in [erased_in_part, *erased_whole, successor]:
        assert delete_fact_authorized(engine, fact_id, trusted_actor_id=_actor(),
                                      source_agent_id="test").get("ok")
    live_fact = list(live.final_fact_ids)[0]
    db_path = engine._db.db_path
    pid = engine._profile_id

    conn = sqlite3.connect(db_path)
    ids = {
        # a memory with one erased fact and survivors: its preview is the memory's opening
        "partial": _event(conn, "memory.stored", {
            "operation_id": partial.operation_id, "fact_ids": partial_facts,
            "path": "canonical_materializer", "content_preview": _TEXT[:60]}),
        # a memory whose every fact was erased
        "whole": _event(conn, "memory.stored", {
            "operation_id": whole.operation_id, "fact_ids": erased_whole,
            "path": "canonical_materializer",
            "content_preview": "Zorblax guards the lonely synthetic bridge"}),
        # an edit event: the preview is the NEW version, which was erased later
        "edit": _event(conn, "memory.updated", {
            "fact_id": live_fact, "successor_fact_id": successor, "status": "proposed",
            "profile_id": pid, "content_preview": "Brennick now keeps the synthetic key"}),
        # a memory nothing was erased from
        "untouched": _event(conn, "memory.stored", {
            "operation_id": live.operation_id, "fact_ids": [live_fact],
            "path": "canonical_materializer",
            "content_preview": "Quillon waters the synthetic fern"}),
    }
    conn.commit()
    conn.close()
    engine.close()
    return {"db": db_path, "events": ids, "texts": ["Marrowick", "Zorblax", "Brennick"]}


def _payload(db_path, event_id: int) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return json.loads(conn.execute("SELECT payload FROM memory_events WHERE id = ?",
                                       (event_id,)).fetchone()[0])
    finally:
        conn.close()


def _scan(db_path) -> dict:
    from superlocalmemory.storage.integrity_scan import plan

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return plan(conn, foreign_keys=False)["erased_text"]
    finally:
        conn.close()


def test_the_scan_counts_the_previews_of_erased_memories(old_events):
    # three events name an erased fact; the fourth names only a live one
    assert _scan(old_events["db"])["event_previews"] == 3


def test_repair_blanks_them_keeps_the_event_and_reports_zero_afterwards(old_events):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    summary = Repair(old_events["db"], limits=Limits(pause_s=0, confirm_s=0)).apply()

    assert summary["status"] == "finished"
    assert summary["before"]["erased_text"]["event_previews"] == 3
    assert summary["after"]["erased_text"]["event_previews"] == 0
    assert summary["done"]["erased_text.copies_scrubbed"] >= 3
    ids = old_events["events"]
    for name in ("partial", "whole", "edit"):
        payload = _payload(old_events["db"], ids[name])
        assert "content_preview" not in payload, name
    # The event itself stays: ids, path, status. Only the words go.
    assert _payload(old_events["db"], ids["partial"])["path"] == "canonical_materializer"
    assert _payload(old_events["db"], ids["edit"])["status"] == "proposed"
    # A memory nothing was erased from keeps its preview.
    assert _payload(old_events["db"], ids["untouched"])["content_preview"].startswith("Quillon")


def test_the_receipt_counts_the_previews_and_holds_no_words(old_events):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    summary = Repair(old_events["db"], limits=Limits(pause_s=0, confirm_s=0)).apply()

    conn = sqlite3.connect(old_events["db"])
    rows = conn.execute("SELECT after_json, undoable FROM integrity_repair_receipts WHERE "
                        "run_id = ? AND action = 'scrub_erased_text'",
                        (summary["run_id"],)).fetchall()
    removed = sum(json.loads(after).get("event_previews_removed", 0) for after, _ in rows)
    assert removed == 3
    assert all(undoable == 0 for _, undoable in rows)
    for word in old_events["texts"]:
        for table, column in (("integrity_repair_receipts", "after_json"),
                              ("integrity_repair_receipts", "before_json"),
                              ("integrity_repair_receipts", "reason"),
                              ("integrity_repair_undo", "row_json"),
                              ("integrity_repair_runs", "summary_json")):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE instr({column}, ?)",  # noqa: S608
                                (word,)).fetchone()[0] == 0, (table, word)
    conn.close()


def test_undo_never_brings_a_preview_back(old_events):
    """A repair never brings back anything that was erased: the scrub keeps no
    copy, so undoing the run leaves the previews blank."""
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    repair = Repair(old_events["db"], limits=Limits(pause_s=0, confirm_s=0))
    summary = repair.apply()
    restored = repair.undo(summary["run_id"])

    assert "memory_events" not in restored
    assert _scan(old_events["db"])["event_previews"] == 0
    for name in ("partial", "whole", "edit"):
        assert "content_preview" not in _payload(old_events["db"], old_events["events"][name])


def test_a_second_repair_finds_nothing_left_to_scrub(old_events):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    Repair(old_events["db"], limits=Limits(pause_s=0, confirm_s=0)).apply()
    again = Repair(old_events["db"], limits=Limits(pause_s=0, confirm_s=0)).apply()

    assert again["done"].get("erased_text.copies_scrubbed", 0) == 0


def test_a_new_erasure_blanks_the_previews_of_a_memory_that_keeps_other_facts(
        engine_with_mock_deps):
    """The same rule at erasure time, so no new leftovers are made."""
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    receipt = _store(engine, _TEXT)
    facts = list(receipt.final_fact_ids)
    if len(facts) < 2:
        pytest.skip("extraction produced one fact; a partial erasure needs two")
    marker = uuid.uuid4().hex[:6]
    with engine._db.raw_connection() as conn:
        event_id = _event(conn, "memory.stored", {
            "operation_id": receipt.operation_id, "fact_ids": facts,
            "content_preview": f"{marker} {_TEXT[:40]}"})
        conn.commit()

    assert delete_fact_authorized(engine, facts[0], trusted_actor_id=_actor(),
                                  source_agent_id="test").get("ok")

    row = engine._db.execute("SELECT payload FROM memory_events WHERE id = ?", (event_id,))[0]
    assert "content_preview" not in json.loads(dict(row)["payload"])


def test_an_erasure_blanks_observe_and_auto_capture_previews_of_the_erased_memory(
        engine_with_mock_deps):
    """Observe and auto-capture events name no fact and no operation; they carry
    the hash of the text they saw, which is the erased memory's source hash.
    Both shapes are planted exactly as the daemon and the MCP observe tool write
    them, for an erased memory and for a live one."""
    import hashlib

    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    erased_text = "Vellatrix hides the synthetic ledger under the third stair."
    live_text = "Ombrel feeds the synthetic heron at dawn."
    erased = _store(engine, erased_text)
    _store(engine, live_text)

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    with engine._db.raw_connection() as conn:
        ids = {
            "observed": _event(conn, "memory.observed", {
                "content_hash": digest(erased_text), "content_preview": erased_text[:120],
                "buffer_size": 1}),
            "auto_observe": _event(conn, "memory.captured", {
                "agent_id": "claude", "category": "decision", "source": "auto-observe",
                "content_hash": digest(erased_text), "content_preview": erased_text[:80]}),
            "live_observed": _event(conn, "memory.observed", {
                "content_hash": digest(live_text), "content_preview": live_text[:120],
                "buffer_size": 1}),
        }
        conn.commit()

    for fact_id in erased.final_fact_ids:
        assert delete_fact_authorized(engine, fact_id, trusted_actor_id=_actor(),
                                      source_agent_id="test").get("ok")

    def payload(event_id: int) -> dict:
        row = engine._db.execute("SELECT payload FROM memory_events WHERE id = ?", (event_id,))[0]
        return json.loads(dict(row)["payload"])

    assert "content_preview" not in payload(ids["observed"])
    assert "content_preview" not in payload(ids["auto_observe"])
    assert payload(ids["auto_observe"])["source"] == "auto-observe"
    assert payload(ids["live_observed"])["content_preview"].startswith("Ombrel")


def test_the_mcp_observe_tool_names_the_text_it_captured_by_hash():
    """Without the hash an auto-observe event names nothing an erasure can find."""
    import inspect

    from superlocalmemory.mcp import tools_active

    source = inspect.getsource(tools_active)
    start = source.index('_emit_event("memory.captured", {')
    emitted = source[start:source.index("}", start)]
    assert '"content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()' in emitted
