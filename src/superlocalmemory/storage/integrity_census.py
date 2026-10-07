# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What is wrong between the tables of a store, by class. Read-only.

Every class says what it is and what ``slm db repair`` does with it:

* ``remove`` — a row derived from a parent that no longer exists, recomputable
  and carrying no history (a keyword token list, a vector map, a retention
  score). The repair removes it, with a receipt and an undo copy (no undo copy
  when the parent was erased: keeping one would keep the erased words).
* ``keep`` — history or content that must not be thrown away because its
  parent went (access log, validity history, entity names and summaries,
  date events that still index live facts, quarantined summaries). Reported.

On a real 22k-fact store every one of the 6,972 foreign-key findings came from
past cleanups that ran with foreign keys off (a 5,201-merge deduplication, a
stop-word entity purge) plus 413 summaries quarantined on purpose. None needs
a parent invented or a constraint dropped, and none of that is done here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

REMOVE = "remove"
KEEP = "keep"


@dataclass(frozen=True)
class OrphanClass:
    table: str
    column: str
    parent: str
    parent_column: str
    action: str
    why: str


ORPHAN_CLASSES: tuple[OrphanClass, ...] = (
    OrphanClass("polar_embeddings", "fact_id", "atomic_facts", "fact_id", REMOVE,
                "compressed vector code of a fact that no longer exists"),
    OrphanClass("embedding_quantization_metadata", "fact_id", "atomic_facts", "fact_id",
                REMOVE, "compression note of a fact that no longer exists"),
    OrphanClass("fact_retention", "fact_id", "atomic_facts", "fact_id", REMOVE,
                "retention score of a fact that no longer exists"),
    OrphanClass("bm25_tokens", "fact_id", "atomic_facts", "fact_id", REMOVE,
                "keyword tokens of a fact that no longer exists (still loaded by keyword search)"),
    OrphanClass("vector_row_map", "fact_id", "atomic_facts", "fact_id", REMOVE,
                "vector link of a fact that no longer exists"),
    OrphanClass("embedding_metadata", "fact_id", "atomic_facts", "fact_id", REMOVE,
                "vector link of a fact that no longer exists"),
    OrphanClass("fact_outcome_score", "fact_id", "atomic_facts", "fact_id", REMOVE,
                "usefulness score of a fact that no longer exists"),
    OrphanClass("entity_communities", "entity_id", "canonical_entities", "entity_id", REMOVE,
                "cluster label of an entity that no longer exists"),
    OrphanClass("fact_access_log", "fact_id", "atomic_facts", "fact_id", KEEP,
                "access history: kept"),
    OrphanClass("fact_temporal_validity", "fact_id", "atomic_facts", "fact_id", KEEP,
                "validity history: kept"),
    OrphanClass("entity_aliases", "entity_id", "canonical_entities", "entity_id", KEEP,
                "names of a removed entity: content, kept and reported"),
    OrphanClass("entity_profiles", "entity_id", "canonical_entities", "entity_id", KEEP,
                "summary of a removed entity: content, kept and reported"),
    OrphanClass("temporal_events", "entity_id", "canonical_entities", "entity_id", KEEP,
                "date events: they still index live facts"),
)


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is not None


def _orphan_sql(k: OrphanClass, select: str) -> str:
    return (f"SELECT {select} FROM {k.table} AS c WHERE c.{k.column} IS NOT NULL "  # noqa: S608
            f"AND c.{k.column} != '' AND NOT EXISTS (SELECT 1 FROM {k.parent} AS p "
            f"WHERE p.{k.parent_column} = c.{k.column})")


def orphan_count(conn: sqlite3.Connection, k: OrphanClass) -> int:
    if not (_has(conn, k.table) and _has(conn, k.parent)):
        return 0
    return int(conn.execute(_orphan_sql(k, "COUNT(*)")).fetchone()[0])


def orphan_rowids(conn: sqlite3.Connection, k: OrphanClass, limit: int) -> list[int]:
    if not (_has(conn, k.table) and _has(conn, k.parent)):
        return []
    rows = conn.execute(_orphan_sql(k, "c.rowid") + " ORDER BY c.rowid LIMIT ?", (limit,))
    return [int(r[0]) for r in rows]


def still_orphan_clause(k: OrphanClass) -> str:
    """Re-checked inside the repair's own transaction: a parent that came back
    since the census keeps its row."""
    return (f"{k.column} IS NOT NULL AND {k.column} != '' AND NOT EXISTS (SELECT 1 FROM "
            f"{k.parent} AS p WHERE p.{k.parent_column} = {k.table}.{k.column})")


def foreign_key_findings(conn: sqlite3.Connection) -> dict[str, int]:
    """``PRAGMA foreign_key_check`` grouped by ``child -> parent``."""
    out: dict[str, int] = {}
    for row in conn.execute("PRAGMA foreign_key_check"):
        key = f"{row[0]} -> {row[2]}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def quarantined_parentless_facts(conn: sqlite3.Connection) -> dict[str, int]:
    """Facts without a memory: summaries kept and quarantined on purpose."""
    if not _has(conn, "atomic_facts"):
        return {"total": 0, "quarantined": 0}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
    quarantined = "SUM(CASE WHEN f.quarantined = 1 THEN 1 ELSE 0 END)" if "quarantined" in cols else "0"
    row = conn.execute(
        f"SELECT COUNT(*), {quarantined} FROM atomic_facts AS f WHERE NOT EXISTS "  # noqa: S608
        "(SELECT 1 FROM memories AS m WHERE m.memory_id = f.memory_id)").fetchone()
    return {"total": int(row[0] or 0), "quarantined": int(row[1] or 0)}


def orphan_census(conn: sqlite3.Connection) -> list[dict]:
    return [{"table": k.table, "parent": k.parent, "action": k.action, "why": k.why,
             "rows": orphan_count(conn, k)} for k in ORPHAN_CLASSES]


__all__ = ["KEEP", "ORPHAN_CLASSES", "OrphanClass", "REMOVE", "foreign_key_findings",
           "orphan_census", "orphan_count", "orphan_rowids", "quarantined_parentless_facts",
           "still_orphan_clause"]
