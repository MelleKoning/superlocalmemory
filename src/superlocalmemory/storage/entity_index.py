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
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

#: The backfill's row in ``fact_entity_association_repair_state``.
REPAIR_KEY = "entity-index-coverage"

#: Fact/entity pairs written per backfill transaction. Each costs about 0.1 ms
#: (two indexes and the foreign keys), so a batch holds the write lock for
#: about 50 ms and a remember waiting behind it barely notices. Bounding by
#: facts alone did not do that: facts name up to ~300 entities, and on the
#: author's store one 250-fact batch was 65,000 pairs and held the lock 6.6 s.
MAX_PAIRS_PER_BATCH = 500

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

    def _write() -> None:
        _insert_pairs(db.execute, fact_id, profile_id, entities, "store")

    # Joins the caller's transaction when there is one (store_fact), so the
    # fact and its index rows commit together; otherwise one of its own.
    db._atomically(_write)


def record_rewritten_fact(db: DatabaseManager, fact_id: str,
                          profile_id: str | None, raw_entities: object) -> None:
    """Index a fact whose entity list an update just replaced.

    A fact that now names more entities must be findable under them; the row
    of an entity it no longer names goes stale and the lookup's JSON test
    filters it. ``profile_id`` None means the caller did not scope the update.
    """
    owner = db.execute("SELECT profile_id FROM atomic_facts WHERE fact_id=?",
                       (fact_id,))
    if not owner:
        return
    fact_profile = str(dict(owner[0])["profile_id"])
    if profile_id is not None and fact_profile != profile_id:
        return
    record_fact_entities(db, fact_id, fact_profile, entity_ids(raw_entities))


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


def _backfill_batch(conn: sqlite3.Connection, batch_size: int,
                    max_pairs: int) -> dict[str, Any]:
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
    inserted = pairs = done = 0
    for rowid, fact_id, profile_id, raw in rows:
        entities = entity_ids(raw)
        if done and pairs + len(entities) > max_pairs:
            break  # the rest is the next batch's; one fact always fits
        inserted += _insert_pairs(conn.execute, fact_id, profile_id,
                                  entities, "coverage-backfill")
        pairs += len(entities)
        done += 1
    conn.execute(
        "UPDATE fact_entity_association_repair_state SET state='running',"
        "last_fact_rowid=?,scanned=scanned+?,inserted=inserted+?,"
        "last_error='',updated_at=? WHERE repair_key=?",
        (int(rows[done - 1][0]), done, inserted, _now(), REPAIR_KEY),
    )
    return {"scanned": done, "inserted": inserted, "complete": False}


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
            result = _backfill_batch(conn, batch_size, MAX_PAIRS_PER_BATCH)
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


@contextmanager
def lookup_session(db: DatabaseManager) -> Iterator[sqlite3.Connection | None]:
    """One read-only snapshot for a run of lookups, or None to use ``db``.

    A bridge walk makes up to ``MAX_ENTITY_LOOKUPS`` lookups, and each one
    through ``db.execute`` opens a fresh connection with a cold page cache;
    the facts of popular entities overlap heavily, so one connection reads
    them once (measured 121 -> 49 ms per lookup on popular entities). The
    read transaction also gives every lookup of the walk the same snapshot.
    A store without a file behind it (a test double) gets None.
    """
    from superlocalmemory.storage.memory_write import memory_read
    from superlocalmemory.storage.read_connection import ReadConnectionError

    path = getattr(db, "db_path", None)
    if not isinstance(path, (str, Path)) or not Path(path).is_file():
        yield None
        return
    with ExitStack() as stack:
        try:
            conn = stack.enter_context(memory_read(path))
            conn.execute("BEGIN")
        except (OSError, sqlite3.Error, ReadConnectionError) as exc:
            # A snapshot that cannot open costs speed, not answers.
            logger.debug("entity lookups fall back to per-call reads: %s", exc)
            conn = None
        try:
            yield conn
        finally:
            if conn is not None:
                conn.rollback()


def _rows(db: DatabaseManager, conn: sqlite3.Connection | None,
          sql: str, args: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    if conn is not None:
        return [tuple(r) for r in conn.execute(sql, args).fetchall()]
    return [tuple(r) for r in db.execute(sql, args)]


def _owner_profile(db: DatabaseManager, entity_id: str,
                   conn: sqlite3.Connection | None) -> str | None:
    rows = _rows(db, conn, "SELECT profile_id FROM canonical_entities "
                           "WHERE entity_id=?", (entity_id,))
    return str(rows[0][0]) if rows else None


def facts_for_entity(
    db: DatabaseManager, entity_id: str, profile_id: str, *, limit: int,
    indexed: bool, include_global: bool = False, include_shared: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    """The newest ``limit`` visible facts naming ``entity_id``, with their entities.

    Same answer either way: the facts in scope whose ``canonical_entities_json``
    holds the id, newest first, ``fact_id`` breaking a tie so the order never
    depends on how SQLite happens to scan. ``indexed`` permits the index (about
    0.3 ms) over the full scan (17-81 ms on the author's store), and it is used
    only where it is provably complete: the entity belongs to this profile and
    only this profile's facts are visible. Every write indexes a fact under the
    entities of its own profile, so a fact naming another profile's entity, a
    deleted entity, or a global or shared fact is reachable only by the scan,
    and gets it. The JSON test stays on the indexed path too, so an association
    a later rewrite of the fact made stale can never produce a match the scan
    would not.
    """
    from superlocalmemory.storage.database import _scope_where

    where, params = _scope_where(profile_id, include_global=include_global,
                                 include_shared=include_shared, prefix="af")
    needle = f'%"{entity_id}"%'
    use_index = (
        indexed and not include_global and not include_shared
        and _owner_profile(db, entity_id, conn) == profile_id
    )
    if use_index:
        sql = (
            "SELECT af.fact_id AS fact_id, af.canonical_entities_json AS ents "
            "FROM fact_entity_associations AS fea "
            "JOIN atomic_facts AS af ON af.fact_id = fea.fact_id "
            "WHERE fea.profile_id = ? AND fea.entity_id = ? "
            f"AND {where} AND af.canonical_entities_json LIKE ? "
            "ORDER BY af.created_at DESC, af.fact_id LIMIT ?"
        )
        args: tuple[Any, ...] = (profile_id, entity_id, *params, needle, int(limit))
    else:
        sql = (
            "SELECT af.fact_id AS fact_id, af.canonical_entities_json AS ents "
            f"FROM atomic_facts AS af WHERE {where} "
            "AND af.canonical_entities_json LIKE ? "
            "ORDER BY af.created_at DESC, af.fact_id LIMIT ?"
        )
        args = (*params, needle, int(limit))
    return [(str(fid), entity_ids(ents)) for fid, ents in _rows(db, conn, sql, args)]


__all__ = ["REPAIR_KEY", "backfill", "entity_ids", "facts_for_entity",
           "is_complete", "lookup_session", "record_fact_entities", "record_rewritten_fact", "status"]
