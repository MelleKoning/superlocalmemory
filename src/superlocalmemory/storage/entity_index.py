# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which facts mention an entity, answered from an index.

WHY. ``fact_entity_associations`` (M028) is indexed on
``(profile_id, entity_id, fact_id)``, but it was only ever filled for the facts
that existed when M028 ran and for facts stored with an ingestion operation id.
Everything else was missing: on the author's store it held 28,465 of the
408,331 fact/entity pairs, every missing one written after M028's boundary.
Callers therefore asked ``canonical_entities_json LIKE '%"id"%'``, a scan of
the whole facts table that cost 17-81 ms per entity, and bridge discovery
bounded those scans with a wall-clock deadline, so its answer depended on how
busy the machine was.

THIS MODULE
* keeps the index complete for every new write (``record_fact_entities``,
  called by ``DatabaseManager.store_fact`` in the fact's own transaction);
* fills in every existing fact with an idempotent background backfill
  (``backfill``), resumable from a durable cursor, run by the daemon after the
  M028 backfill. It is not a migration: it touches only these two tables, in
  short batches, so no restore point of memory.db is taken for it;
* answers lookups from the index once that backfill is complete, and from the
  old scan until then (``facts_for_entity``), so results never get worse while
  the backfill runs.

Every row written here has ``count_applied = 0``: it records that the pair
exists, never that an entity's ``fact_count`` was incremented. The counting
path in ``core.store_pipeline`` claims a row by flipping it to 1, which it does
exactly as it would for a row it inserted itself, so counts are unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

#: The backfill's row in ``fact_entity_association_repair_state``.
REPAIR_KEY = "entity-index-coverage"

_INSERT_PAIR = (
    "INSERT OR IGNORE INTO fact_entity_associations "
    "(profile_id,fact_id,entity_id,first_operation_id,count_applied) "
    "SELECT ?,?,?,?,0 FROM canonical_entities "
    "WHERE profile_id=? AND entity_id=?"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def entity_ids(raw: object) -> tuple[str, ...]:
    """Distinct entity ids from a ``canonical_entities_json`` value."""
    try:
        values = json.loads(str(raw or "[]"))
    except (TypeError, ValueError):
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(dict.fromkeys(str(value) for value in values if value))


def _insert_pairs(execute, fact_id: str, profile_id: str,
                  entities: tuple[str, ...], source: str) -> int:
    inserted = 0
    for entity_id in entities:
        result = execute(
            _INSERT_PAIR,
            (profile_id, fact_id, entity_id, source, profile_id, entity_id),
        )
        inserted += max(0, getattr(result, "rowcount", 0) or 0)
    return inserted


def _has_index_table(db: DatabaseManager) -> bool:
    """Whether M028's table exists. True is cached; a store opened before its
    migrations ran is probed again on the next write."""
    if getattr(db, "_entity_index_table_present", False):
        return True
    present = bool(db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='fact_entity_associations'",
    ))
    if present:
        db._entity_index_table_present = True
    return present


def record_fact_entities(db: DatabaseManager, fact_id: str, profile_id: str,
                         canonical_entities: list[str] | tuple[str, ...]) -> None:
    """Index a just-written fact's entities. Idempotent; never counts.

    An entity id without a ``canonical_entities`` row is skipped, exactly as
    the M028 backfill skips it: the foreign key would refuse it. A store whose
    migrations have not created the table yet is left alone; the backfill
    indexes this fact once they have.
    """
    entities = tuple(dict.fromkeys(str(e) for e in canonical_entities if e))
    if not entities or not _has_index_table(db):
        return
    for entity_id in entities:
        db.execute(
            _INSERT_PAIR,
            (profile_id, fact_id, entity_id, "store", profile_id, entity_id),
        )


# -- background backfill ---------------------------------------------------------

def _ensure_state(conn: sqlite3.Connection) -> None:
    target = int(conn.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM atomic_facts",
    ).fetchone()[0])
    conn.execute(
        "INSERT OR IGNORE INTO fact_entity_association_repair_state "
        "(repair_key,state,target_fact_rowid,updated_at) VALUES (?,?,?,?)",
        (REPAIR_KEY, "pending", target, _now()),
    )


def _backfill_batch(conn: sqlite3.Connection, batch_size: int) -> dict[str, Any]:
    _ensure_state(conn)
    cursor, target = conn.execute(
        "SELECT last_fact_rowid,target_fact_rowid "
        "FROM fact_entity_association_repair_state WHERE repair_key=?",
        (REPAIR_KEY,),
    ).fetchone()
    rows = conn.execute(
        "SELECT rowid,fact_id,profile_id,canonical_entities_json "
        "FROM atomic_facts WHERE rowid>? AND rowid<=? ORDER BY rowid LIMIT ?",
        (int(cursor), int(target), batch_size),
    ).fetchall()
    if not rows:
        conn.execute(
            "UPDATE fact_entity_association_repair_state "
            "SET state='complete',last_error='',updated_at=? WHERE repair_key=?",
            (_now(), REPAIR_KEY),
        )
        return {"scanned": 0, "inserted": 0, "complete": True}
    inserted = 0
    for rowid, fact_id, profile_id, raw in rows:
        inserted += _insert_pairs(conn.execute, fact_id, profile_id,
                                  entity_ids(raw), "coverage-backfill")
    conn.execute(
        "UPDATE fact_entity_association_repair_state SET state='running',"
        "last_fact_rowid=?,scanned=scanned+?,inserted=inserted+?,"
        "last_error='',updated_at=? WHERE repair_key=?",
        (int(rows[-1][0]), len(rows), inserted, _now(), REPAIR_KEY),
    )
    return {"scanned": len(rows), "inserted": inserted, "complete": False}


def backfill(db_path: Path, *, batch_size: int = 250,
             max_batches: int = 1) -> dict[str, Any]:
    """Run up to ``max_batches`` short backfill batches. Safe to repeat.

    Each batch is its own write transaction under ``memory_write`` (the
    process write lock), so a store in use keeps answering between batches.
    """
    from superlocalmemory.storage.memory_write import memory_write

    if batch_size < 1 or max_batches < 1:
        raise ValueError("batch_size and max_batches must be positive")
    totals: dict[str, Any] = {"scanned": 0, "inserted": 0, "complete": False}
    for _ in range(max_batches):
        with memory_write(Path(db_path)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            result = _backfill_batch(conn, batch_size)
        totals["scanned"] += result["scanned"]
        totals["inserted"] += result["inserted"]
        totals["complete"] = result["complete"]
        if result["complete"]:
            break
    return totals


def status(db_path: Path) -> dict[str, Any]:
    """Durable backfill progress, for the daemon's status report."""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        row = conn.execute(
            "SELECT state,target_fact_rowid,last_fact_rowid,scanned,inserted,"
            "last_error,updated_at FROM fact_entity_association_repair_state "
            "WHERE repair_key=?", (REPAIR_KEY,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        conn.close()
    keys = ("state", "target_fact_rowid", "last_fact_rowid", "scanned",
            "inserted", "last_error", "updated_at")
    if row is None:
        return {"state": "pending", "target_fact_rowid": 0, "last_fact_rowid": 0,
                "scanned": 0, "inserted": 0, "last_error": "", "updated_at": ""}
    return dict(zip(keys, row))


# -- lookups ---------------------------------------------------------------------

def is_complete(db: DatabaseManager) -> bool:
    """True once every fact written before the backfill began is indexed."""
    try:
        rows = db.execute(
            "SELECT state FROM fact_entity_association_repair_state "
            "WHERE repair_key=?", (REPAIR_KEY,),
        )
    except sqlite3.Error:
        return False
    return bool(rows) and str(dict(rows[0])["state"]) == "complete"


def facts_for_entity(
    db: DatabaseManager, entity_id: str, profile_id: str, *, limit: int,
    indexed: bool, include_global: bool = False, include_shared: bool = False,
) -> list[tuple[str, tuple[str, ...]]]:
    """The newest ``limit`` visible facts naming ``entity_id``, with their entities.

    Same answer either way: the facts in scope whose ``canonical_entities_json``
    holds the id, newest first, ``fact_id`` breaking a tie so the order never
    depends on how SQLite happens to scan. ``indexed`` picks the index (about
    0.3 ms) over the full scan (17-81 ms on the author's store). The JSON test
    stays on the indexed path too, so an association a later rewrite of the
    fact made stale can never produce a match the scan would not.
    """
    from superlocalmemory.storage.database import _scope_where

    where, params = _scope_where(profile_id, include_global=include_global,
                                 include_shared=include_shared, prefix="af")
    needle = f'%"{entity_id}"%'
    if indexed:
        profiles = [profile_id]
        if include_global or include_shared:
            # Global and shared facts belong to other profiles; seek every
            # profile that has an association, read from the index itself.
            profiles = sorted({profile_id, *(
                str(dict(r)["profile_id"]) for r in db.execute(
                    "SELECT DISTINCT profile_id FROM fact_entity_associations")
            )})
        marks = ",".join("?" * len(profiles))
        sql = (
            "SELECT af.fact_id AS fact_id, af.canonical_entities_json AS ents "
            "FROM fact_entity_associations AS fea "
            "JOIN atomic_facts AS af ON af.fact_id = fea.fact_id "
            f"WHERE fea.profile_id IN ({marks}) AND fea.entity_id = ? "
            f"AND {where} AND af.canonical_entities_json LIKE ? "
            "ORDER BY af.created_at DESC, af.fact_id LIMIT ?"
        )
        args: tuple[Any, ...] = (*profiles, entity_id, *params, needle, int(limit))
    else:
        sql = (
            "SELECT af.fact_id AS fact_id, af.canonical_entities_json AS ents "
            f"FROM atomic_facts AS af WHERE {where} "
            "AND af.canonical_entities_json LIKE ? "
            "ORDER BY af.created_at DESC, af.fact_id LIMIT ?"
        )
        args = (*params, needle, int(limit))
    return [
        (str(dict(r)["fact_id"]), entity_ids(dict(r)["ents"]))
        for r in db.execute(sql, args)
    ]


__all__ = ["REPAIR_KEY", "backfill", "entity_ids", "facts_for_entity",
           "is_complete", "record_fact_entities", "status"]
