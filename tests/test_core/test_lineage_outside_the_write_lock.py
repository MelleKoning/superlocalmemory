# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A remember's lineage is worked out before its checkpoint takes the write lock.

On a copy of a 22k-fact store the final checkpoint of every remember read all
486,822 graph edges, 11,591 scenes, 3,436 summaries and 30,064 lexical rows of
the profile INSIDE its write transaction (2.6-4.4 s of write lock per remember,
the main source of save and edit tail waits). Now the reads are indexed to the
operation's own facts and run first, with no lock; the transaction only writes.
The withholding of a fact that turned a number into a date still commits
atomically with its lineage and the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.core import derivation_lineage as lineage
from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M019_derivation_lineage,
)

SOURCE = "The Kestrel checkpoint was recalled at rank 1 in 2004.6 ms."
DATE_FACT = "The recall happened on June 1st, 2004"


def _db(path: Path) -> DatabaseManager:
    db = DatabaseManager(path)
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M019_derivation_lineage.apply(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
        if "quarantined" not in columns:
            conn.execute("ALTER TABLE atomic_facts ADD COLUMN quarantined INTEGER DEFAULT 0")
    return db


def _command(db: DatabaseManager, facts: dict[str, str], materialize=None,
             key: str = "lock-1") -> tuple[IngestionCommand, str]:
    def write_queryable(request: IngestionRequest, _operation_id: str) -> list[str]:
        db.execute("INSERT INTO memories (memory_id,profile_id,content,session_id,speaker,role) "
                   "VALUES (?,?,?, '', '', 'user')", (f"m-{key}", request.profile_id,
                                                      request.content))
        for fact_id, content in facts.items():
            db.execute("INSERT INTO atomic_facts (fact_id,memory_id,profile_id,content,fact_type)"
                       " VALUES (?,?,?,?,'semantic')", (fact_id, f"m-{key}",
                                                        request.profile_id, content))
        return list(facts)

    command = IngestionCommand(
        IngestionOperationRepository(db), write_queryable=write_queryable,
        materialize=materialize or (lambda operation: operation.queryable_fact_ids),
        derivation_version="lock-test")
    receipt = command.submit(IngestionRequest(content=SOURCE, profile_id="default",
                                              source_type="test", idempotency_key=key))
    return command, receipt.operation_id


def _in_transaction(db: DatabaseManager) -> bool:
    return getattr(db._txn_state, "conn", None) is not None  # noqa: SLF001


def test_lineage_is_worked_out_before_the_checkpoint_transaction(tmp_path, monkeypatch) -> None:
    db = _db(tmp_path / "memory.db")
    real_plan = lineage.plan_operation_lineage
    planned_inside: list[bool] = []

    def watched_plan(db_arg, **kw):
        planned_inside.append(_in_transaction(db_arg))
        return real_plan(db_arg, **kw)

    monkeypatch.setattr(lineage, "plan_operation_lineage", watched_plan)
    command, op_id = _command(db, {"f-verbatim": SOURCE, "f-date": DATE_FACT})
    assert command.materialize(op_id).state.value == "complete"
    assert planned_inside and not any(planned_inside)  # never under the write lock


def test_a_fact_that_became_a_date_is_still_withheld_with_its_lineage(tmp_path) -> None:
    db = _db(tmp_path / "memory.db")
    command, op_id = _command(db, {"f-verbatim": SOURCE, "f-date": DATE_FACT})
    assert command.materialize(op_id).state.value == "complete"
    rows = {r["fact_id"]: dict(r) for r in db.execute(
        "SELECT f.fact_id, COALESCE(f.quarantined,0) AS q, l.unresolved_reason AS r "
        "FROM atomic_facts f JOIN derivation_lineage l "
        "ON l.object_type='fact' AND l.object_id=f.fact_id")}
    assert rows["f-date"]["q"] == 1
    assert rows["f-date"]["r"].startswith("source_fidelity:")
    assert rows["f-verbatim"]["q"] == 0


def test_a_withholding_rolls_back_with_its_checkpoint(tmp_path, monkeypatch) -> None:
    """Atomic: a checkpoint that fails leaves the fact as it was, not half-withheld."""
    db = _db(tmp_path / "memory.db")
    real_record = lineage._record
    calls = {"n": 0}

    def failing_record(db_arg, **kw):
        calls["n"] += 1
        if kw.get("object_type") == "profile":
            raise RuntimeError("disk full")
        return real_record(db_arg, **kw)

    monkeypatch.setattr(lineage, "_record", failing_record)
    command, op_id = _command(db, {"f-verbatim": SOURCE, "f-date": DATE_FACT})
    assert command.materialize(op_id).state.value == "failed"
    q = db.execute("SELECT COALESCE(quarantined,0) AS q FROM atomic_facts WHERE fact_id='f-date'")
    assert q[0]["q"] == 0
    assert db.execute("SELECT COUNT(*) AS n FROM derivation_lineage")[0]["n"] == 0


def test_an_edit_between_the_plan_and_the_write_is_never_overwritten(tmp_path,
                                                                     monkeypatch) -> None:
    """The plan is checked against the facts inside the transaction and redone if stale."""
    db = _db(tmp_path / "memory.db")
    real_plan = lineage.plan_operation_lineage
    edited = {"done": False}

    def plan_then_edit(db_arg, **kw):
        plan = real_plan(db_arg, **kw)
        if not edited["done"] and "f-date" in kw["fact_ids"]:
            edited["done"] = True  # the user corrects the fact before the write lands
            db.execute("UPDATE atomic_facts SET content=? WHERE fact_id='f-date'",
                       ("The Kestrel checkpoint was recalled at rank 1",))
        return plan

    monkeypatch.setattr(lineage, "plan_operation_lineage", plan_then_edit)
    command, op_id = _command(db, {"f-verbatim": SOURCE, "f-date": DATE_FACT})
    assert command.materialize(op_id).state.value == "complete"
    row = db.execute(
        "SELECT COALESCE(f.quarantined,0) AS q, l.source_status AS s FROM atomic_facts f "
        "JOIN derivation_lineage l ON l.object_type='fact' AND l.object_id=f.fact_id "
        "WHERE f.fact_id='f-date'")[0]
    assert (row["q"], row["s"]) == (0, "exact")  # what the fact says NOW


# -- the indexed reads find exactly what the full scans found -------------------


def _legacy_records(db, *, operation_id, profile_id, raw_content, fact_ids, version):
    """The 4.1.21 algorithm (full-profile scans), kept here as the reference."""
    def fact_ids_of(value):
        try:
            decoded = json.loads(value or "[]")
        except (TypeError, ValueError):
            return ()
        return tuple(str(i) for i in decoded) if isinstance(decoded, list) else ()

    ops = frozenset(fact_ids)
    out = set()
    for fid in fact_ids:
        rows = db.execute("SELECT content FROM atomic_facts WHERE fact_id=? AND profile_id=?",
                          (fid, profile_id))
        content = str(rows[0]["content"]) if rows else None
        status = "unresolved" if content is None or raw_content.find(content) < 0 else "exact"
        out.add(("fact", fid, status, ()))
        if status == "exact":
            out.add(("index_bm25", fid, "exact", (fid,)))
        else:
            out.add(("index_bm25", fid, "unresolved", (fid,)))
    out.add(("profile", profile_id, "not_applicable", ()))
    for table, col, kind in (("entity_profiles", "profile_entry_id", "entity_summary"),
                             ("memory_scenes", "scene_id", "memory_scene")):
        for row in db.execute(f"SELECT {col},fact_ids_json FROM {table} WHERE profile_id=?",
                              (profile_id,)):
            ids = tuple(f for f in fact_ids_of(row["fact_ids_json"]) if f in ops)
            if ids:
                out.add((kind, str(row[col]), "derived_from_facts", ids))
    for row in db.execute("SELECT edge_id,source_id,target_id FROM graph_edges "
                          "WHERE profile_id=?", (profile_id,)):
        ids = tuple(v for v in (str(row["source_id"]), str(row["target_id"])) if v in ops)
        if ids:
            out.add(("graph_edge", str(row["edge_id"]), "derived_from_facts", ids))
    indexed = {str(r["fact_id"]) for r in db.execute(
        "SELECT fact_id FROM bm25_tokens WHERE profile_id=?", (profile_id,))}
    return {r for r in out if r[0] != "index_bm25" or r[1] in indexed}


@pytest.mark.parametrize("n_unrelated", [0, 700])
def test_indexed_reads_find_exactly_what_the_full_scans_found(tmp_path, n_unrelated) -> None:
    db = _db(tmp_path / "memory.db")
    with db.transaction():  # one commit for the whole fixture
        db.execute("INSERT INTO profiles (profile_id, name) VALUES ('other', 'Other')")
        db.execute("INSERT INTO memories (memory_id,profile_id,content) VALUES ('m1','default','s')")
        db.execute("INSERT INTO memories (memory_id,profile_id,content) VALUES ('m2','other','s')")
        op_facts = tuple(f"op-{i}" for i in range(450))  # more than one query chunk
        for fid in op_facts + ("x-1", "x-2"):
            db.execute("INSERT INTO atomic_facts (fact_id,memory_id,profile_id,content,fact_type) "
                       "VALUES (?,'m1','default',?,'semantic')",
                       (fid, "Alpha uses SQLite" if fid.endswith("0") else f"paraphrase {fid}"))
        db.execute("INSERT INTO canonical_entities (entity_id,profile_id,canonical_name) "
                   "VALUES ('e1','default','Alpha')")
        db.execute("INSERT INTO canonical_entities (entity_id,profile_id,canonical_name) "
                   "VALUES ('e2','other','Beta')")
        summaries = {"ep-hit": json.dumps(["op-3", "x-1"]), "ep-miss": json.dumps(["x-1"]),
                     "ep-bad": "{not json op-3", "ep-obj": json.dumps({"a": "op-3"}),
                     "ep-substr": json.dumps(["op-31x"]), "ep-num": json.dumps([7, "op-449"])}
        for i, (pid, ids) in enumerate(summaries.items()):
            db.execute("INSERT INTO canonical_entities (entity_id,profile_id,canonical_name) "
                       "VALUES (?,'default',?)", (f"e-{i}", f"Entity {i}"))
            db.execute("INSERT INTO entity_profiles (profile_entry_id,entity_id,profile_id,"
                       "fact_ids_json) VALUES (?,?,'default',?)", (pid, f"e-{i}", ids))
        db.execute("INSERT INTO entity_profiles (profile_entry_id,entity_id,profile_id,fact_ids_json)"
                   " VALUES ('ep-other','e2','other',?)", (json.dumps(["op-3"]),))
        for sid, ids in (("s-hit", ["op-10", "op-11"]), ("s-miss", ["x-2"])):
            db.execute("INSERT INTO memory_scenes (scene_id,profile_id,fact_ids_json) "
                       "VALUES (?,'default',?)", (sid, json.dumps(ids)))
        edges = [("g-src", "op-5", "e1"), ("g-tgt", "e1", "op-6"), ("g-both", "op-7", "op-8"),
                 ("g-miss", "x-1", "e1")]
        edges += [(f"g-u{i}", f"x-{i % 2 + 1}", "e1") for i in range(n_unrelated)]
        for eid, src, tgt in edges:
            db.execute("INSERT INTO graph_edges (edge_id,profile_id,source_id,target_id,edge_type) "
                       "VALUES (?,'default',?,?,'entity')", (eid, src, tgt))
        db.execute("INSERT INTO graph_edges (edge_id,profile_id,source_id,target_id,edge_type) "
                   "VALUES ('g-other','other','op-5','e2','entity')")
        for fid in ("op-0", "op-400", "x-1"):
            db.execute("INSERT INTO bm25_tokens (fact_id,profile_id,tokens) VALUES (?,'default','[]')",
                       (fid,))

    kw = dict(operation_id="op", profile_id="default", raw_content="Alpha uses SQLite",
              fact_ids=op_facts + ("op-missing",))
    plan = lineage.plan_operation_lineage(db, derivation_version="v", **kw)
    got = {(r["object_type"], r["object_id"], r["source_status"],
            tuple(r.get("source_fact_ids", ()))) for r in plan.records}
    assert got == _legacy_records(db, version="v", **kw)
    assert ("entity_summary", "ep-num", "derived_from_facts", ("op-449",)) in got


def test_a_second_checkpoint_of_the_same_facts_rewrites_nothing(tmp_path, monkeypatch) -> None:
    """Every materialization checkpoints twice with the same facts; on a large
    store the second rewrote 500-1,300 identical rows under the lock."""
    db = _db(tmp_path / "memory.db")
    command, op_id = _command(db, {"f-verbatim": SOURCE, "f-date": DATE_FACT})
    assert command.materialize(op_id).state.value == "complete"
    op = command.repository.get(op_id)
    before = {r["lineage_id"]: dict(r) for r in db.execute("SELECT * FROM derivation_lineage")}
    writes: list[str] = []
    real_record = lineage._record
    monkeypatch.setattr(lineage, "_record",
                        lambda db_arg, **kw: (writes.append(kw["object_type"]),
                                              real_record(db_arg, **kw)))
    plan = lineage.lineage_plan_for(db, op, op.final_fact_ids, op.derivation_version)
    with db.transaction():
        lineage.apply_operation_lineage(db, plan)
    assert writes == []
    assert {r["lineage_id"]: dict(r) for r in db.execute("SELECT * FROM derivation_lineage")} \
        == before
    db.execute("UPDATE derivation_lineage SET source_status='unresolved' "
               "WHERE object_type='fact' AND object_id='f-verbatim'")
    with db.transaction():  # a row that differs IS written again
        lineage.apply_operation_lineage(db, plan)
    assert writes == ["fact"]
