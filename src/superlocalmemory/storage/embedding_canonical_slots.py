# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The canonical vector columns, swapped by name instead of rewritten.

``atomic_facts.embedding / fisher_mean / fisher_variance`` are part of the
live space. Rewriting them for every fact inside the swap took 6.2 s of a
7.3 s transaction on a 22,284-fact store (rows average 9.5 KB), while every
request waited. So each has a ``*_next`` twin: ``ADD COLUMN`` is O(1), the job
fills the twin batch by batch, and the swap is three ``RENAME COLUMN``
statements per column (32 ms measured on that store) -- the old values end in
``*_next`` and are cleared in the background afterwards.

A rename fires no trigger and rewrites the triggers that name the column, so
:func:`swap` re-creates the vector change-log trigger and logs a vector change
for every fact, which every change-tracked cache reads.
"""

from __future__ import annotations

from typing import Any

CANONICAL = ("embedding", "fisher_mean", "fisher_variance")
NEXT_SUFFIX = "_next"
_SWAP_SUFFIX = "_swap"
CLEAR_CHUNK = 256


def _columns(conn: Any) -> set[str]:
    return {str(r[1]) for r in conn.execute("PRAGMA table_info(atomic_facts)")}


def live_columns(conn: Any) -> list[str]:
    present = _columns(conn)
    return [c for c in CANONICAL if c in present]


def next_name(column: str) -> str:
    return column + NEXT_SUFFIX


def ensure_next_columns(conn: Any) -> list[str]:
    """Add the ``*_next`` twins that are missing (O(1) each). Returns the live names."""
    present = _columns(conn)
    live = [c for c in CANONICAL if c in present]
    for column in live:
        if next_name(column) not in present:
            conn.execute(f"ALTER TABLE atomic_facts ADD COLUMN {next_name(column)} BLOB")
    return live


def write_next(conn: Any, fact_id: str, values: dict[str, bytes | None]) -> None:
    live = [c for c in CANONICAL if c in values]
    sets = ", ".join(f"{next_name(c)} = ?" for c in live)
    conn.execute(f"UPDATE atomic_facts SET {sets} WHERE fact_id = ?",
                 (*(values[c] for c in live), fact_id))


def _recreate_vector_trigger(conn: Any) -> None:
    from superlocalmemory.storage import fact_search_changes as fsc

    present = _columns(conn)
    watched = tuple(c for c in fsc.VECTOR_COLUMNS if c in present)
    conn.execute(f"DROP TRIGGER IF EXISTS {fsc.trigger_name('update_vector')}")
    conn.execute(fsc._trigger(
        "update_vector", f"AFTER UPDATE OF {', '.join(watched)}",
        "SELECT NEW.profile_id, NEW.fact_id, 'v' UNION SELECT OLD.profile_id, OLD.fact_id, 'v'"))


def _log_every_vector_changed(conn: Any) -> None:
    from superlocalmemory.storage import fact_search_changes as fsc

    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (fsc.TABLE,)).fetchone():
        return
    conn.execute(f"INSERT INTO {fsc.TABLE} (profile_id, fact_id, what) "
                 "SELECT profile_id, fact_id, 'v' FROM atomic_facts")
    conn.execute(f"DELETE FROM {fsc.TABLE} WHERE seq <= "
                 f"(SELECT MAX(seq) FROM {fsc.TABLE}) - {fsc.RETAINED_CHANGES}")


def swap(conn: Any) -> list[str]:
    """Exchange every live column with its twin. Caller holds the write txn."""
    swapped = []
    present = _columns(conn)
    for column in CANONICAL:
        twin = next_name(column)
        if column not in present or twin not in present:
            continue
        conn.execute(f"ALTER TABLE atomic_facts RENAME COLUMN {column} TO {column}{_SWAP_SUFFIX}")
        conn.execute(f"ALTER TABLE atomic_facts RENAME COLUMN {twin} TO {column}")
        conn.execute(f"ALTER TABLE atomic_facts RENAME COLUMN {column}{_SWAP_SUFFIX} TO {twin}")
        swapped.append(column)
    _recreate_vector_trigger(conn)
    _log_every_vector_changed(conn)
    return swapped


def missing_next(conn: Any, column: str = "embedding") -> int:
    """Facts whose twin is still empty (a row replaced after it was staged)."""
    twin = next_name(column)
    if twin not in _columns(conn):
        return -1
    return int(conn.execute(f"SELECT COUNT(*) FROM atomic_facts WHERE {twin} IS NULL")
               .fetchone()[0])


def clear_next_chunk(conn: Any, after_rowid: int) -> int | None:
    """NULL the twins of the next CLEAR_CHUNK rows; the new cursor, None when done."""
    twins = [next_name(c) for c in CANONICAL if next_name(c) in _columns(conn)]
    if not twins:
        return None
    rows = conn.execute("SELECT rowid FROM atomic_facts WHERE rowid > ? ORDER BY rowid LIMIT ?",
                        (after_rowid, CLEAR_CHUNK)).fetchall()
    if not rows:
        return None
    last = int(rows[-1][0])
    sets = ", ".join(f"{t} = NULL" for t in twins)
    where = " OR ".join(f"{t} IS NOT NULL" for t in twins)
    conn.execute(f"UPDATE atomic_facts SET {sets} WHERE rowid > ? AND rowid <= ? AND ({where})",
                 (after_rowid, last))
    return last


__all__ = ["CANONICAL", "CLEAR_CHUNK", "clear_next_chunk", "ensure_next_columns",
           "live_columns", "missing_next", "next_name", "swap", "write_next"]
