# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""learning.db across a restore (L1-05).

Two rules, the same as for the memory store:

 * An erased profile stays erased. Its learning rows (the facts it was shown,
   its feedback, its agent receipts) are removed from the STAGING copy of
   learning.db before that copy is written, so they never reach the live file.
 * Nothing is lost silently. Learning recorded after the copy is NOT carried
   over, and the preview says how much.

Why not carry it over: learning.db is mostly aggregates and models --
``bandit_arms`` sums, ``learning_model_state`` blobs, engagement counters.
Merging the live rows into the copy's would count the shared history twice and
pair a model with signals it was not trained on. Row-level merging is safe only
for append-only logs, which are a minority of the file, and a half-merged
learning state is worse than a coherent older one. The memories themselves are
untouched by this choice; learning rebuilds from use.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

logger = logging.getLogger(__name__)

#: Ledger rows that record a profile was closed; an erasure keeps them.
_LEDGERS = frozenset({"agent_receipt_profile_closures"})
_BOOKKEEPING = frozenset({"migration_log", "slm_schema_version", "sqlite_sequence"})


def _regular_tables(conn: sqlite3.Connection, schema: str = "main") -> list[str]:
    rows = conn.execute(
        f"SELECT name, sql FROM {schema}.sqlite_master WHERE type='table'").fetchall()
    virtual = {n for n, sql in rows if (sql or "").upper().startswith("CREATE VIRTUAL")}
    return [n for n, _sql in rows
            if n not in virtual and not n.startswith("sqlite_")
            and not any(n.startswith(v + "_") for v in virtual)]


def _has_profile_column(conn: sqlite3.Connection, table: str) -> bool:
    return any(r[1] == "profile_id" for r in conn.execute(f"PRAGMA table_info('{table}')"))


def profiles_in(learning_db: Path) -> set[str]:
    """Every profile id that has a row in ``learning_db``."""
    found: set[str] = set()
    with closing(sqlite3.connect(str(learning_db))) as conn:
        for table in _regular_tables(conn):
            if _has_profile_column(conn, table):
                found |= {r[0] for r in conn.execute(
                    f"SELECT DISTINCT profile_id FROM {table}") if r[0]}
    return found


def replay_erasures(staged_learning: Path, profiles: set[str]) -> int:
    """Remove every row of ``profiles`` from the staging copy. Returns rows removed."""
    if not profiles:
        return 0
    removed = 0
    with closing(sqlite3.connect(str(staged_learning), isolation_level=None)) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table in _regular_tables(conn):
                if table in _LEDGERS or not _has_profile_column(conn, table):
                    continue
                for pid in sorted(profiles):
                    cur = conn.execute(f"DELETE FROM {table} WHERE profile_id=?", (pid,))
                    removed += max(cur.rowcount, 0)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("the prepared learning copy failed its check")
    return removed


def records_since(learning_db: Path, learning_snapshot: Path | None) -> int | None:
    """Rows in the live learning.db that the copy does not have (by rowid).

    None when either file cannot be read. Reads only (``mode=ro`` / immutable).
    """
    if learning_snapshot is None or not Path(learning_db).exists():
        return 0
    try:
        with closing(sqlite3.connect(f"{Path(learning_db).absolute().as_uri()}?mode=ro",
                                     uri=True)) as conn:
            conn.execute("ATTACH DATABASE ? AS snap",
                         (f"{Path(learning_snapshot).absolute().as_uri()}"
                          "?mode=ro&immutable=1",))
            in_copy = set(_regular_tables(conn, "snap"))
            total = 0
            for table in _regular_tables(conn):
                if table in _BOOKKEEPING:
                    continue
                if table not in in_copy:
                    sql = f"SELECT COUNT(*) FROM main.{table}"
                else:
                    sql = (f"SELECT COUNT(*) FROM main.{table} WHERE rowid NOT IN "
                           f"(SELECT rowid FROM snap.{table})")
                try:
                    total += int(conn.execute(sql).fetchone()[0] or 0)
                except sqlite3.OperationalError:
                    continue                     # a WITHOUT ROWID table: not counted
            return total
    except sqlite3.Error as exc:
        logger.warning("[SLM] Could not compare learning.db with the copy: %s", exc)
        return None


def lost_learning_warning(count: int | None) -> str | None:
    if count is None:
        return ("Your current learning data could not be read, so it is replaced by the "
                "copy's. Your memories are not affected.")
    if count <= 0:
        return None
    return (f"{count:,} learning records made after the copy (feedback, ranking signals) "
            "are not carried over: learning goes back to the copy's state. Your memories "
            "are not affected.")


__all__ = ["lost_learning_warning", "profiles_in", "records_since", "replay_erasures"]
