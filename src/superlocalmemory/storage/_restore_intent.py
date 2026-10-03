# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Which rows missing from the live store were deleted ON PURPOSE.

A restore must repeat a deletion a person asked for and must undo a loss. The
product already records every deliberate deletion before it happens:

  projection_tombstones  one row per forgotten or erased fact, written by the
                         erasure service before the canonical delete (forget,
                         ``delete_memory``, entity and profile erasure alike),
                         never pruned (see ``retention_policy``)
  erasure_receipts       a ``profile`` receipt in state COMPLETE: the whole
                         profile was erased (it is replayed as a profile wipe)

So a fact of the copy that is missing from the live store is a deliberate
deletion exactly when a tombstone names it -- in the live store or in the copy
itself (an erasure that was part-way through when the copy was taken) -- or
when its profile was erased. Anything else was lost, and the copy brings it back.

A memory row follows its facts: it was deleted on purpose when a tombstone
names it, or names one of its facts in the copy, or its profile was erased. The
staging delete keeps any memory a returning fact still points at.

Every fragment reads ``main`` (the live store) and ``snap`` (the copy) through
the read-only comparison connection; nothing here writes.
"""

from __future__ import annotations

import sqlite3

_TOMBSTONES = "projection_tombstones"
_RECEIPTS = "erasure_receipts"


def _has(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    return conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,)).fetchone() is not None


def _tombstone_schemas(conn: sqlite3.Connection) -> list[str]:
    return [s for s in ("main", "snap") if _has(conn, s, _TOMBSTONES)]


def _any(clauses: list[str]) -> str:
    return "(" + " OR ".join(clauses) + ")" if clauses else "0"


def erased_profile(conn: sqlite3.Connection, alias: str) -> str:
    """True for rows of a profile whose erasure completed in the live store."""
    if not _has(conn, "main", _RECEIPTS):
        return "0"
    return (f"EXISTS (SELECT 1 FROM main.{_RECEIPTS} r WHERE r.subject_type='profile' "
            f"AND r.state='COMPLETE' AND r.profile_id = {alias}.profile_id)")


def fact_intended(conn: sqlite3.Connection, alias: str = "s") -> str:
    """True when a recorded forget/erasure covers the copy's fact ``alias``."""
    return _any([
        f"EXISTS (SELECT 1 FROM {schema}.{_TOMBSTONES} t WHERE t.fact_id = {alias}.fact_id)"
        for schema in _tombstone_schemas(conn)])


def memory_intended(conn: sqlite3.Connection, alias: str = "s") -> str:
    """True when a recorded forget/erasure covers the copy's memory ``alias``."""
    clauses: list[str] = []
    for schema in _tombstone_schemas(conn):
        clauses.append(f"EXISTS (SELECT 1 FROM {schema}.{_TOMBSTONES} t "
                       f"WHERE t.memory_id = {alias}.memory_id)")
        clauses.append(f"EXISTS (SELECT 1 FROM snap.atomic_facts sf JOIN "
                       f"{schema}.{_TOMBSTONES} t ON t.fact_id = sf.fact_id "
                       f"WHERE sf.memory_id = {alias}.memory_id)")
    return _any(clauses)


_FACT_MISSING = ("FROM snap.atomic_facts s WHERE NOT EXISTS "
                 "(SELECT 1 FROM main.atomic_facts f WHERE f.fact_id = s.fact_id)")
_MEMORY_MISSING = (
    "FROM snap.memories s WHERE NOT EXISTS "
    "(SELECT 1 FROM main.memories m WHERE m.memory_id = s.memory_id) "
    "AND NOT EXISTS (SELECT 1 FROM snap.atomic_facts sf JOIN main.atomic_facts lf "
    "ON lf.fact_id = sf.fact_id WHERE sf.memory_id = s.memory_id)")


def deleted_facts_sql(conn: sqlite3.Connection) -> str:
    """``FROM ...`` of the copy's facts deleted on purpose since the copy."""
    return f"{_FACT_MISSING} AND {fact_intended(conn)}"


def deleted_memories_sql(conn: sqlite3.Connection) -> str:
    return f"{_MEMORY_MISSING} AND {memory_intended(conn)}"


def returning_facts_sql(conn: sqlite3.Connection) -> str:
    """The copy's facts missing now with no recorded deletion: they come back."""
    return (f"{_FACT_MISSING} AND NOT {fact_intended(conn)} "
            f"AND NOT {erased_profile(conn, 's')}")


def returning_memories_sql(conn: sqlite3.Connection) -> str:
    return (f"FROM snap.memories s WHERE NOT EXISTS "
            f"(SELECT 1 FROM main.memories m WHERE m.memory_id = s.memory_id) "
            f"AND NOT {memory_intended(conn)} AND NOT {erased_profile(conn, 's')}")


__all__ = ["deleted_facts_sql", "deleted_memories_sql", "erased_profile", "fact_intended",
           "memory_intended", "returning_facts_sql", "returning_memories_sql"]
