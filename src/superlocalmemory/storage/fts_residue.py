# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Deleted words must leave the keyword index, not just stop matching.

An FTS5 delete only records a "deleted" marker; the term itself stays in the
index segments until they are merged. Measured through a real daemon: after a
memory was fully erased, ``atomic_facts_fts_data`` still held its unique word,
readable by anyone who opens the file, though no query could match it.

Two fixes, both idempotent:

* ``ensure_secure_delete``: FTS5's ``secure-delete`` option (SQLite 3.42+)
  makes every later delete remove the term from the index at once. It is a
  persistent setting stored in the index itself, so it covers every connection
  and process. On an older SQLite the option is unknown; that is reported, not
  raised, and the repair's ``optimize`` remains the way to purge.
* ``purge_deleted_terms``: ``optimize`` rewrites the index without the words of
  rows deleted before ``secure-delete`` was on (the existing-store repair).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The keyword indexes that hold memory text.
FTS_TABLES: tuple[str, ...] = ("atomic_facts_fts", "fact_expansion_fts")


def _run(target: Any, sql: str, params: tuple = ()) -> list:
    result = target.execute(sql, params)
    return list(result.fetchall() if hasattr(result, "fetchall") else result)


def _exists(target: Any, table: str) -> bool:
    return bool(_run(target, "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)))


def secure_delete_on(target: Any, table: str) -> bool:
    """True when the index already removes deleted terms at once."""
    rows = _run(target, f"SELECT v FROM {table}_config WHERE k = 'secure-delete'")  # noqa: S608
    return bool(rows) and str(tuple(rows[0])[0]) == "1"


def ensure_secure_delete(target: Any) -> dict[str, str]:
    """Turn ``secure-delete`` on for every keyword index. Cheap when already on.

    ``target`` is a ``sqlite3.Connection`` or a ``DatabaseManager``. Returns
    ``{table: "on" | "enabled" | "absent" | "unsupported"}``.
    """
    state: dict[str, str] = {}
    for table in FTS_TABLES:
        if not _exists(target, table):
            state[table] = "absent"
            continue
        if secure_delete_on(target, table):
            state[table] = "on"
            continue
        try:
            _run(target, f"INSERT INTO {table}({table}, rank) "  # noqa: S608
                 "VALUES('secure-delete', 1)")
            state[table] = "enabled"
        except Exception as exc:  # SQLite older than 3.42
            if "secure-delete" not in str(exc).lower() and "unknown" not in str(exc).lower():
                raise
            logger.warning("keyword index %s cannot purge deleted words at once: %s",
                           table, exc)
            state[table] = "unsupported"
    return state


def enable_quietly(conn: Any) -> None:
    """At store creation: never stops a store from opening."""
    try:
        ensure_secure_delete(conn)
    except Exception as exc:
        logger.warning("keyword index secure delete not enabled: %s", exc)


def purge_deleted_terms(conn: Any, table: str) -> None:
    """Rewrite one index without the words of already-deleted rows."""
    _run(conn, f"INSERT INTO {table}({table}) VALUES('optimize')")  # noqa: S608


__all__ = ["FTS_TABLES", "enable_quietly", "ensure_secure_delete", "purge_deleted_terms", "secure_delete_on"]
