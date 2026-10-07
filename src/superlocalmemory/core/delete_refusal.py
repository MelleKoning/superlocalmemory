# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Put a memory's search entries back when its delete is refused at the last check.

A delete asks whether correction history protects the fact, removes the fact's
search entries (keyword tokens, live keyword index, time validity, vector, ANN)
and writes its erasure tombstone, then deletes the fact in one transaction that
asks again. A person may propose a correction naming the fact in between. The
last check refuses, as it must (the history is kept on purpose), and before
4.1.22 the memory stayed stored but could no longer be found, and the tombstone
kept every later re-index from repairing it.

The entries cannot share the last check's transaction: the vector index lives
on its own connection. So :func:`snapshot` records them before they go, and
:func:`restore` puts back exactly what was there, removes the tombstone and the
erasure's open obligations, and leaves the refusal's "Nothing was changed" true.
Only a definite refusal is undone: a writer that timed out may still delete, and
undoing then would bring a deleted memory's entries back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Rows keyed by the fact that the erasure removes and a restore copies back.
_KEYED_ROWS: tuple[str, ...] = ("bm25_tokens", "fact_temporal_validity")
#: Removed directly only when no vector store is loaded (see ``VectorOwner``).
_VECTOR_ROWS: tuple[str, ...] = ("embedding_metadata", "vector_row_map")


@dataclass(frozen=True)
class SearchSnapshot:
    profile_id: str
    fact_id: str
    rows: dict[str, tuple[dict, ...]] = field(default_factory=dict)
    in_live_index: bool = False
    in_vector_store: bool = False
    in_ann: bool = False


def _missing_table(exc: Exception) -> bool:
    return "no such table" in str(exc).lower()


def _rows(db: Any, table: str, fact_id: str) -> tuple[dict, ...]:
    try:
        return tuple(dict(r) for r in db.execute(
            f"SELECT * FROM {table} WHERE fact_id = ?", (fact_id,)))  # noqa: S608
    except Exception as exc:
        if _missing_table(exc):
            return ()
        raise


def _bm25(engine: Any) -> Any:
    return getattr(getattr(engine, "_retrieval_engine", None), "_bm25", None)


def _store_available(engine: Any) -> bool:
    store = getattr(engine, "_vector_store", None)
    return store is not None and bool(getattr(store, "available", False))


def snapshot(engine: Any, profile_id: str, fact_id: str) -> SearchSnapshot:
    """What a search reaches the fact through. Call before the erasure removes it."""
    db = engine._db
    tables = _KEYED_ROWS + (() if _store_available(engine) else _VECTOR_ROWS)
    index, ann = _bm25(engine), getattr(engine, "_ann_index", None)
    in_store = False
    if _store_available(engine):
        in_store = fact_id in engine._vector_store.indexed_fact_ids(profile_id)
    return SearchSnapshot(
        profile_id=profile_id, fact_id=fact_id,
        rows={t: _rows(db, t, fact_id) for t in tables},
        in_live_index=index is not None and fact_id in getattr(index, "_fact_id_set", ()),
        in_vector_store=in_store,
        in_ann=ann is not None and hasattr(ann, "contains") and bool(ann.contains(fact_id)),
    )


def _put_back(db: Any, table: str, rows: tuple[dict, ...], fact_id: str) -> None:
    if not rows or _rows(db, table, fact_id):
        return  # nothing was there, or it is back already: never duplicate
    for row in rows:
        cols = list(row)
        db.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES "  # noqa: S608
                   f"({', '.join('?' for _ in cols)})", tuple(row[c] for c in cols))


def _close_erasure(db: Any, snap: SearchSnapshot, operation_id: str) -> None:
    for sql, args in (
        ("DELETE FROM projection_tombstones WHERE profile_id = ? AND fact_id = ?",
         (snap.profile_id, snap.fact_id)),
        ("DELETE FROM projection_obligations WHERE operation_id = ? AND kind = 'erase'",
         (operation_id,)),
    ):
        try:
            db.execute(sql, args)
        except Exception as exc:
            if not _missing_table(exc):
                raise


def _embedding(db: Any, snap: SearchSnapshot) -> list[float] | None:
    from superlocalmemory.storage.embedding_codec import decode_embedding

    rows = db.execute("SELECT embedding FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                      (snap.fact_id, snap.profile_id))
    raw = dict(rows[0]).get("embedding") if rows else None
    return decode_embedding(raw, fact_id=snap.fact_id) if raw else None


def _restore_live(engine: Any, snap: SearchSnapshot) -> None:
    db = engine._db
    content_rows = db.execute("SELECT content FROM atomic_facts WHERE fact_id = ? "
                              "AND profile_id = ?", (snap.fact_id, snap.profile_id))
    index = _bm25(engine)
    if snap.in_live_index and index is not None and content_rows:
        index.update_fact(snap.fact_id, dict(content_rows[0])["content"], snap.profile_id)
    if not (snap.in_vector_store or snap.in_ann):
        return
    embedding = _embedding(db, snap)
    if not embedding:
        return
    if snap.in_vector_store and _store_available(engine):
        engine._vector_store.upsert(fact_id=snap.fact_id, profile_id=snap.profile_id,
                                    embedding=embedding)
    ann = getattr(engine, "_ann_index", None)
    if snap.in_ann and ann is not None and hasattr(ann, "add"):
        ann.add(snap.fact_id, embedding)


def restore(engine: Any, snap: SearchSnapshot, operation_id: str) -> bool:
    """Undo the erasure's removals for a fact that is still stored.

    Returns False (and does nothing) when the fact is gone: then the erasure is
    real and its tombstone must stay. Never raises: the caller is reporting a
    refusal, and a restore failure is logged so it is not silent.
    """
    db = engine._db
    try:
        with db.transaction():
            if not db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ? "
                              "LIMIT 1", (snap.fact_id, snap.profile_id)):
                return False
            _close_erasure(db, snap, operation_id)
            for table, rows in snap.rows.items():
                _put_back(db, table, rows, snap.fact_id)
        _restore_live(engine, snap)
        return True
    except Exception as exc:
        logger.error("Refused delete of %s: search entries not fully restored: %s",
                     snap.fact_id[:16], exc)
        return False


__all__ = ["SearchSnapshot", "restore", "snapshot"]
