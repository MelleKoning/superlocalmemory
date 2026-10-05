# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Read-only presence and inventory queries for ``VectorStore``.

Split out of ``vector_store`` to keep that module within the file-size limit.
``VectorStore`` exposes each of these as a method of the same name.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.retrieval.vector_store import VectorStore

logger = logging.getLogger("superlocalmemory.retrieval.vector_store")


def raw_vector_present(store: VectorStore, fact_id: str) -> bool:
    if not store._available:
        try:
            conn = sqlite3.connect(str(store._db_path))
            try:
                row = conn.execute(
                    "SELECT 1 FROM vector_row_map WHERE fact_id = ? LIMIT 1",
                    (fact_id,),
                ).fetchone()
                return row is not None
            finally:
                conn.close()
        except Exception:
            return False
    try:
        with store._managed_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM vector_row_map vrm "
                "WHERE vrm.fact_id = ? "
                "AND EXISTS (SELECT 1 FROM fact_embeddings fe "
                "WHERE fe.rowid = vrm.vec_rowid)",
                (fact_id,),
            ).fetchone()
            return row is not None
    except Exception:
        try:
            conn2 = sqlite3.connect(str(store._db_path))
            try:
                row = conn2.execute(
                    "SELECT 1 FROM vector_row_map WHERE fact_id = ? LIMIT 1",
                    (fact_id,),
                ).fetchone()
                return row is not None
            finally:
                conn2.close()
        except Exception:
            return False


def is_searchable_by_meaning(
    store: VectorStore,
    fact_id: str,
    profile_id: str | None = None,
) -> bool:
    """Return True if search() would be able to return this fact.

    The check mirrors search()'s own join:

        FROM fact_embeddings AS fe
        JOIN embedding_metadata AS em
          ON em.vec_rowid = fe.rowid
         AND em.profile_id = fe.profile_id

    A fact that has a vector in fact_embeddings but no row in
    embedding_metadata will NOT be returned by search(), so this method
    returns False for it — even though raw_vector_present() would return
    True.  Callers that need to decide whether a fact requires re-embedding
    must use this method, not raw_vector_present().

    Returns False on any error or when the store is unavailable (fail-
    closed: never optimistic).
    """
    if not store._available:
        return False
    try:
        with store._managed_connection() as conn:
            if profile_id is not None:
                sql = (
                    "SELECT 1 FROM fact_embeddings AS fe "
                    "JOIN embedding_metadata AS em "
                    "ON em.vec_rowid = fe.rowid "
                    "AND em.profile_id = fe.profile_id "
                    "WHERE em.fact_id = ? "
                    "AND fe.profile_id = ? "
                    "LIMIT 1"
                )
                row = conn.execute(sql, (fact_id, profile_id)).fetchone()
            else:
                sql = (
                    "SELECT 1 FROM fact_embeddings AS fe "
                    "JOIN embedding_metadata AS em "
                    "ON em.vec_rowid = fe.rowid "
                    "AND em.profile_id = fe.profile_id "
                    "WHERE em.fact_id = ? "
                    "LIMIT 1"
                )
                row = conn.execute(sql, (fact_id,)).fetchone()
            return row is not None
    except Exception:
        return False


def count(store: VectorStore, profile_id: str | None = None) -> int:
    """Count complete metadata/vector pairs in the store.

    Returns 0 if unavailable.
    """
    if not store._available:
        return 0

    try:
        with store._managed_connection() as conn:
            if profile_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) AS c "
                    "FROM embedding_metadata em "
                    "JOIN fact_embeddings fe "
                    "ON fe.rowid = em.vec_rowid "
                    "AND fe.profile_id = em.profile_id "
                    "WHERE em.profile_id = ?",
                    (profile_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c "
                    "FROM embedding_metadata em "
                    "JOIN fact_embeddings fe "
                    "ON fe.rowid = em.vec_rowid "
                    "AND fe.profile_id = em.profile_id",
                ).fetchone()
        return int(row["c"]) if row else 0
    except Exception as exc:
        logger.debug("count failed: %s", exc)
        return 0


def indexed_fact_ids(store: VectorStore, profile_id: str) -> set[str]:
    """Return fact IDs backed by both metadata and a vec0 payload."""
    if not store._available:
        return set()
    try:
        with store._managed_connection() as conn:
            rows = conn.execute(
                "SELECT em.fact_id "
                "FROM embedding_metadata em "
                "JOIN fact_embeddings fe "
                "ON fe.rowid = em.vec_rowid "
                "AND fe.profile_id = em.profile_id "
                "WHERE em.profile_id = ?",
                (profile_id,),
            ).fetchall()
        return {str(row["fact_id"]) for row in rows}
    except Exception as exc:
        logger.debug("indexed_fact_ids failed: %s", exc)
        return set()
