# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which facts were written or re-worded while a model switch runs.

The swap must prove the staged space still matches every fact. Proving it by
re-reading and hashing all of them inside the swap's transaction took 0.96 s on
a 22k-fact store, and 18.7 s on the same store under machine load, with every
request held meanwhile. So the full comparison runs BEFORE the swap with no
lock held, and the swap only asks this log whether anything changed since that
comparison: O(changes), not O(facts).

The store's own change log (storage/fact_search_changes.py) is not enough: it
does not record an edit of a fact's words that leaves its vector alone, and
such a fact would go live with a vector of its old words. These two triggers
record every insert (including a row replaced in place) and every change of
``content`` or ``profile_id``. Deletions are the erasure trigger's job. They
exist only while a job is active; ids only, never memory text.
"""

from __future__ import annotations

from typing import Any

TABLE = "reembed_changed"
TRIGGERS = ("trg_reembed_fact_inserted", "trg_reembed_fact_reworded")

_DDL = (
    f"CREATE TABLE IF NOT EXISTS {TABLE} (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
    "fact_id TEXT NOT NULL)",
    f"CREATE TRIGGER IF NOT EXISTS {TRIGGERS[0]} AFTER INSERT ON atomic_facts BEGIN "
    f"INSERT INTO {TABLE} (fact_id) VALUES (NEW.fact_id); END",
    f"CREATE TRIGGER IF NOT EXISTS {TRIGGERS[1]} AFTER UPDATE OF content, profile_id "
    f"ON atomic_facts BEGIN INSERT INTO {TABLE} (fact_id) VALUES (NEW.fact_id); END",
)


def start(conn: Any) -> None:
    """Begin recording (idempotent; keeps what a resumed job already logged)."""
    for statement in _DDL:
        conn.execute(statement)


def stop(conn: Any) -> None:
    for trigger in TRIGGERS:
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")


def active(conn: Any) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (TRIGGERS[1],)
                        ).fetchone() is not None


def mark(conn: Any) -> int:
    return int(conn.execute(f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE}").fetchone()[0])


def changed_since(conn: Any, seq: int) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE seq > ?", (seq,)).fetchone()[0])


__all__ = ["TABLE", "TRIGGERS", "active", "changed_since", "mark", "start", "stop"]
