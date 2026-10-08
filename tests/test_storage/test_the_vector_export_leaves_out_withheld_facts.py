# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""A rebuilt vector projection holds the facts recall may return, and no others.

The projection worker keeps withheld (quarantined) and soft-deleted facts out of
Lance: it removes a fact's vector when the fact stops being visible. The bulk
export that builds a projection from scratch (``slm db scale prepare``) read
every vector, so a rebuilt Lance held the withheld ones again. Those vectors
fill nearest-neighbour slots that hydration then discards, and the Lance search
stopped agreeing with the sqlite-vec search it replaces.

Both readers here must answer the same question the worker does, and the count
that the staged-promotion parity check compares against must be the count of
what is exported.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
import sqlite_vec

from superlocalmemory.storage.sqlite_vectors import (
    count_canonical_vectors,
    iter_canonical_vectors,
)


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    enable = getattr(conn, "enable_load_extension", None)
    if not callable(enable):
        pytest.skip("this CPython build does not support SQLite extensions")
    enable(True)
    sqlite_vec.load(conn)
    enable(False)
    conn.executescript(
        """
        CREATE TABLE atomic_facts (
            fact_id TEXT PRIMARY KEY,
            lifecycle TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            quarantined INTEGER NOT NULL DEFAULT 0,
            archive_status TEXT NOT NULL DEFAULT 'live'
        );
        CREATE TABLE embedding_metadata (
            vec_rowid INTEGER PRIMARY KEY,
            fact_id TEXT NOT NULL UNIQUE,
            profile_id TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE fact_embeddings USING vec0(
            profile_id TEXT PARTITION KEY,
            embedding float[8] distance_metric=cosine
        );
        """
    )
    rows = (
        # fact_id, lifecycle, quarantined, archive_status
        ("shown-active", "active", 0, "live"),
        ("shown-cold", "cold", 0, "live"),
        ("withheld", "warm", 1, "live"),
        ("soft-deleted", "warm", 0, "archived"),
    )
    for seed, (fact_id, lifecycle, quarantined, archive) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO atomic_facts VALUES (?, ?, 'default', ?, ?)",
            (fact_id, lifecycle, quarantined, archive),
        )
        cursor = conn.execute(
            "INSERT INTO fact_embeddings(profile_id, embedding) VALUES ('default', ?)",
            (np.full(8, seed, dtype=np.float32).tobytes(),),
        )
        conn.execute(
            "INSERT INTO embedding_metadata (vec_rowid, fact_id, profile_id) "
            "VALUES (?, ?, 'default')",
            (cursor.lastrowid, fact_id),
        )
    conn.commit()
    return conn


def test_the_export_omits_withheld_and_soft_deleted_facts() -> None:
    exported = {row[1] for row in iter_canonical_vectors(_store(), "default")}
    assert exported == {"shown-active", "shown-cold"}


def test_the_count_is_the_number_exported() -> None:
    conn = _store()
    assert count_canonical_vectors(conn, "default") == len(
        list(iter_canonical_vectors(conn, "default"))
    )


def test_a_store_without_the_withholding_columns_exports_every_fact() -> None:
    """Older stores have neither column; nothing there is withheld."""
    conn = sqlite3.connect(":memory:")
    enable = getattr(conn, "enable_load_extension", None)
    if not callable(enable):
        pytest.skip("this CPython build does not support SQLite extensions")
    enable(True)
    sqlite_vec.load(conn)
    enable(False)
    conn.executescript(
        """
        CREATE TABLE atomic_facts (
            fact_id TEXT PRIMARY KEY, lifecycle TEXT NOT NULL, profile_id TEXT NOT NULL
        );
        CREATE TABLE embedding_metadata (
            vec_rowid INTEGER PRIMARY KEY, fact_id TEXT NOT NULL UNIQUE,
            profile_id TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE fact_embeddings USING vec0(
            profile_id TEXT PARTITION KEY, embedding float[8] distance_metric=cosine
        );
        INSERT INTO atomic_facts VALUES ('a', 'active', 'default'), ('b', 'cold', 'default');
        """
    )
    for fact_id in ("a", "b"):
        cursor = conn.execute(
            "INSERT INTO fact_embeddings(profile_id, embedding) VALUES ('default', ?)",
            (np.ones(8, dtype=np.float32).tobytes(),),
        )
        conn.execute(
            "INSERT INTO embedding_metadata VALUES (?, ?, 'default')",
            (cursor.lastrowid, fact_id),
        )
    conn.commit()

    assert {row[1] for row in iter_canonical_vectors(conn, "default")} == {"a", "b"}
    assert count_canonical_vectors(conn, "default") == 2
