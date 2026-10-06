# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A change log of the facts whose searchable vector or kind may have changed.

WHY
---
Recall keeps two things in memory that used to be re-read from the store on
every recall: the vectors (``retrieval/canonical_vector_index``) and each fact's
displayed kind (``retrieval/kind_scope``). Re-reading was the cost: without the
sqlite-vec extension the semantic channel and the spreading-activation seed
search each read every fact with ``SELECT *`` -- 5.6-6.7 s of CPU each on a
21,739-fact store -- and both were abandoned at the 8 s hang guard on every
recall. A cache is only allowed if it can never serve a fact the store no
longer has, and always sees one it has just been given. So each cache needs to
know exactly WHICH facts changed. That is this table.

HOW
---
Triggers, not writer-side calls -- the same reasoning as ``graph_generation``:
facts are written from many places and processes (store, enrichment,
re-embedding, quarantine, soft delete, erasure, scope changes, kind backfill,
restore, migrations), and a trigger sees every one of them.

The UPDATE triggers name their columns, so recall's own access-counter updates
are never logged (they would turn every recall into a refresh). Each row says
what changed: ``v`` (vector, owner, scope or visibility), ``k`` (kind or legacy
type) or ``*`` (inserted or deleted). A kind backfill therefore never makes the
vector index re-read embeddings.

The log keeps the newest :data:`RETAINED_CHANGES` entries; the trigger prunes
older ones itself, so nothing has to maintain it. A reader further behind than
that sees the gap and rebuilds from the store, which is always correct.

Rows hold ids only -- never memory text or vectors -- so erasure leaves no
content here. Applied by ``schema.create_all_tables`` on every engine start,
like ``graph_generation``; no migration needed.
"""

from __future__ import annotations

from typing import Any

TABLE = "fact_search_changes"

#: How many recent changes the log keeps. A reader further behind rebuilds.
RETAINED_CHANGES = 100_000

#: Columns whose change can change what a vector search may return.
VECTOR_COLUMNS: tuple[str, ...] = (
    "fact_id", "profile_id", "embedding", "scope", "shared_with",
    "quarantined", "archive_status",
)
#: Columns whose change can change a fact's displayed kind or its visibility.
KIND_COLUMNS: tuple[str, ...] = (
    "fact_id", "profile_id", "memory_kind", "memory_kind_source",
    "memory_kind_confidence", "fact_type", "scope", "shared_with",
    "quarantined", "archive_status",
)

_PRUNE = (
    f"DELETE FROM {TABLE} WHERE seq <= "
    f"(SELECT MAX(seq) FROM {TABLE}) - {RETAINED_CHANGES};"
)
_EVENTS = ("insert", "delete", "update_vector", "update_kind")


def trigger_name(event: str) -> str:
    return f"trg_atomic_facts_search_change_{event}"


def _trigger(event: str, timing: str, rows: str) -> str:
    return (
        f"CREATE TRIGGER IF NOT EXISTS {trigger_name(event)}\n"
        f"{timing} ON atomic_facts\nBEGIN\n"
        f"    INSERT INTO {TABLE} (profile_id, fact_id, what) {rows};\n"
        f"    {_PRUNE}\nEND;"
    )


def _update_trigger(event: str, columns: tuple[str, ...], what: str) -> str:
    return (
        f"DROP TRIGGER IF EXISTS {trigger_name(event)};\n"
        + _trigger(
            event,
            f"AFTER UPDATE OF {', '.join(columns)}",
            f"SELECT NEW.profile_id, NEW.fact_id, '{what}' "
            f"UNION SELECT OLD.profile_id, OLD.fact_id, '{what}'",
        )
    )


_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id  TEXT NOT NULL,
    fact_id     TEXT NOT NULL,
    what        TEXT NOT NULL
);
{_trigger("insert", "AFTER INSERT", "VALUES (NEW.profile_id, NEW.fact_id, '*')")}
{_trigger("delete", "AFTER DELETE", "VALUES (OLD.profile_id, OLD.fact_id, '*')")}
"""


def ddl(conn: Any) -> str:
    """The log, its triggers, and UPDATE triggers naming the columns present.

    Some watched columns arrive with migrations (``archive_status`` with M011,
    the kind columns with M052), after the first ``create_all_tables`` on a
    fresh store. So the UPDATE triggers are re-created from the columns the
    table has now on every start, and a column added since is watched from the
    next start. Until then the caches stay safe: every channel still
    authorises its candidates through the canonical visibility predicate.
    """
    present = {str(row[1]) for row in conn.execute("PRAGMA table_info(atomic_facts)")}
    if not present:
        return ""
    vector = tuple(c for c in VECTOR_COLUMNS if c in present)
    kind = tuple(c for c in KIND_COLUMNS if c in present)
    return "\n".join((
        _TABLE_DDL,
        _update_trigger("update_vector", vector, "v"),
        _update_trigger("update_kind", kind, "k"),
        "",
    ))


def trigger_names() -> tuple[str, ...]:
    """Every trigger :func:`ddl` creates, for drop and verification."""
    return tuple(trigger_name(e) for e in _EVENTS)


def log_bounds(db: Any) -> tuple[int, int | None]:
    """``(newest seq, oldest retained seq)``; ``(0, None)`` for an empty log.

    Raises whatever the read raises (a store without the log): the caller must
    then not trust any cache keyed on it.
    """
    rows = db.execute(f"SELECT MAX(seq) AS head, MIN(seq) AS oldest FROM {TABLE}")
    row = dict(rows[0]) if rows else {}
    oldest = row.get("oldest")
    return int(row.get("head") or 0), (int(oldest) if oldest is not None else None)


def changed_fact_ids(db: Any, after: int, upto: int, what: str) -> list[str]:
    """Distinct ids changed in ``(after, upto]`` for ``what`` (``v`` or ``k``)."""
    rows = db.execute(
        f"SELECT DISTINCT fact_id FROM {TABLE} "
        "WHERE seq > ? AND seq <= ? AND what IN (?, '*')",
        (after, upto, what),
    )
    return [str(dict(r)["fact_id"]) for r in rows]


def changed_fact_ids_among(db: Any, after: int, fact_ids: tuple[str, ...]) -> list[str]:
    """Which of ``fact_ids`` changed (in any watched way) after ``after``."""
    if not fact_ids:
        return []
    marks = ",".join("?" for _ in fact_ids)
    rows = db.execute(
        f"SELECT DISTINCT fact_id FROM {TABLE} WHERE seq > ? AND fact_id IN ({marks})",
        (after, *fact_ids),
    )
    return [str(dict(r)["fact_id"]) for r in rows]
