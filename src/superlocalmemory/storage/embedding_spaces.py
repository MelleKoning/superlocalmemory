# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Embedding spaces on disk: the live one, the one being built, the previous one.

WHY
---
Changing the embedding model re-embeds every memory. 4.1.21 did that inside
engine start-up while every recall and remember waited (hours on a big store).
4.1.22 builds the new space beside the live one, in the same store, and swaps
them in one short transaction. This module owns the tables that makes possible.

Three facts, each measured on sqlite 3.53 / sqlite-vec 0.1.9, shape it:

* ``ALTER TABLE <vec0> RENAME`` renames only the virtual table, not its shadow
  tables. A staged table renamed that way reads the OLD table's chunks
  ("vectors blob size doesn't match"). So a vec0 table is always renamed
  together with its four shadow tables (:func:`rename_vec`).
* A trigger that touches a vec0 table breaks ``DELETE FROM atomic_facts`` on any
  connection that has not loaded sqlite-vec ("no such module: vec0"). So the
  erasure trigger touches ordinary tables only; vectors it orphans are queued in
  ``reembed_purge`` and removed by :func:`purge_pending`.
* The tables exist only on stores that switch, and are created here with
  ``IF NOT EXISTS`` rather than by a migration: a memory.db migration makes every
  upgrade copy memory.db (about 8 s per GB).

Rows hold ids, hashes and vectors -- never memory text.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

SPACE = "embedding_space"
JOBS = "embedding_reindex_jobs"
LIVE_VEC = "fact_embeddings"
NEXT_VEC = "reembed_next_vec"
NEXT_MAP = "reembed_next_map"
PREV_VEC = "reembed_prev_vec"
PREV_MAP = "reembed_prev_map"
PURGE = "reembed_purge"
TRIGGER = "trg_reembed_fact_erased"
#: The shadow tables sqlite-vec 0.1.x creates for ``vec0(profile_id TEXT
#: PARTITION KEY, embedding float[N])``. :func:`rename_vec` refuses a table
#: whose shadow set differs, rather than renaming part of it.
VEC_SHADOWS = ("chunks", "info", "rowids", "vector_chunks00")

#: Job states. Only these four may still change the store.
_VEC_SHADOW_LIKE = re.compile(
    r"^(chunks|info|rowids|auxiliary|vector_chunks\d+|metadatachunks\d+|metadatatext\d+)$")

ACTIVE_STATES = ("queued", "running", "catching_up", "ready")
TERMINAL_STATES = ("activated", "failed", "rolled_back", "cancelled")

_CONTROL_DDL = (f"""
CREATE TABLE IF NOT EXISTS {SPACE} (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    live_signature   TEXT NOT NULL,
    live_config      TEXT NOT NULL,
    prev_signature   TEXT,
    prev_config      TEXT,
    prev_job_id      INTEGER,
    updated_at       REAL NOT NULL
)""", f"""
CREATE TABLE IF NOT EXISTS {JOBS} (
    job_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL,
    state          TEXT NOT NULL,
    from_signature TEXT NOT NULL,
    to_signature   TEXT NOT NULL,
    from_config    TEXT NOT NULL,
    to_config      TEXT NOT NULL,
    cursor         INTEGER NOT NULL DEFAULT 0,
    next_rowid     INTEGER NOT NULL DEFAULT 0,
    total          INTEGER NOT NULL DEFAULT 0,
    done           INTEGER NOT NULL DEFAULT 0,
    copied         INTEGER NOT NULL DEFAULT 0,
    caught_up      INTEGER NOT NULL DEFAULT 0,
    attempts       INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    stats          TEXT,
    created_at     REAL NOT NULL,
    started_at     REAL,
    updated_at     REAL NOT NULL,
    finished_at    REAL,
    activated_at   REAL
)""")

_SIDE_DDL = (f"""
CREATE TABLE IF NOT EXISTS {NEXT_MAP} (
    fact_id         TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL,
    vec_rowid       INTEGER NOT NULL UNIQUE,
    content_hash    TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    fisher_mean     BLOB,
    fisher_variance BLOB
)""", f"""
CREATE TABLE IF NOT EXISTS {PREV_MAP} (
    fact_id      TEXT PRIMARY KEY,
    profile_id   TEXT NOT NULL,
    vec_rowid    INTEGER NOT NULL,
    content_hash TEXT NOT NULL
)""", f"""
CREATE TABLE IF NOT EXISTS {PURGE} (
    space     TEXT NOT NULL,
    vec_rowid INTEGER NOT NULL,
    PRIMARY KEY (space, vec_rowid)
)""", f"""
CREATE TRIGGER IF NOT EXISTS {TRIGGER} AFTER DELETE ON atomic_facts
BEGIN
    INSERT OR IGNORE INTO {PURGE} (space, vec_rowid)
        SELECT 'next', vec_rowid FROM {NEXT_MAP} WHERE fact_id = OLD.fact_id;
    INSERT OR IGNORE INTO {PURGE} (space, vec_rowid)
        SELECT 'prev', vec_rowid FROM {PREV_MAP} WHERE fact_id = OLD.fact_id;
    DELETE FROM {NEXT_MAP} WHERE fact_id = OLD.fact_id;
    DELETE FROM {PREV_MAP} WHERE fact_id = OLD.fact_id;
END""")


# -- connections -------------------------------------------------------------

def connect(db_path: str | Path) -> sqlite3.Connection:
    """A connection with sqlite-vec loaded; the caller owns transactions."""
    import sqlite_vec

    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except BaseException:
        conn.close()
        raise
    return conn


def table_exists(conn: Any, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1", (name,)
    ).fetchone()
    return row is not None


# -- signatures and key-free configs -----------------------------------------

def signature_of(emb: Any) -> str:
    from superlocalmemory.storage.embedding_migrator import _model_signature

    class _C:  # _model_signature reads config.embedding
        embedding = emb
    return _model_signature(_C)


def same_space(sig_a: str | None, sig_b: str | None) -> bool:
    from superlocalmemory.storage.embedding_migrator import _normalize_signature

    if not sig_a or not sig_b:
        return False
    return sig_a == sig_b or _normalize_signature(sig_a) == _normalize_signature(sig_b)


def public_config(emb: Any) -> dict:
    """An embedding config as JSON-safe data WITHOUT its key."""
    data = asdict(emb)
    data.pop("api_key", None)
    return data


def config_from_public(data: dict | str, api_key: str = "") -> Any:
    from superlocalmemory.core.config import EmbeddingConfig

    raw = json.loads(data) if isinstance(data, str) else dict(data or {})
    known = {f.name for f in fields(EmbeddingConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known and k != "api_key"}
    return EmbeddingConfig(**kwargs, api_key=api_key or "")


# -- control tables ----------------------------------------------------------

def ensure_control_tables(conn: Any) -> None:
    for statement in _CONTROL_DDL:  # one by one: executescript would COMMIT
        conn.execute(statement)


def read_space(conn: Any) -> dict | None:
    if not table_exists(conn, SPACE):
        return None
    row = conn.execute(f"SELECT * FROM {SPACE} WHERE id = 1").fetchone()
    return dict(row) if row is not None else None


def write_space(conn: Any, live_sig: str, live_cfg: dict, prev_sig: str | None = None,
                prev_cfg: dict | None = None, prev_job_id: int | None = None) -> None:
    conn.execute(
        f"INSERT INTO {SPACE} (id, live_signature, live_config, prev_signature, "
        "prev_config, prev_job_id, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET live_signature = excluded.live_signature, "
        "live_config = excluded.live_config, prev_signature = excluded.prev_signature, "
        "prev_config = excluded.prev_config, prev_job_id = excluded.prev_job_id, "
        "updated_at = excluded.updated_at",
        (live_sig, json.dumps(live_cfg, sort_keys=True), prev_sig,
         json.dumps(prev_cfg, sort_keys=True) if prev_cfg is not None else None,
         prev_job_id, time.time()),
    )


# -- vec0 tables -------------------------------------------------------------

def vec_dimension(conn: Any, table: str) -> int | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
    ).fetchone()
    if row is None or not row[0]:
        return None
    match = re.search(r"float\[(\d+)\]", str(row[0]), re.IGNORECASE)
    return int(match.group(1)) if match else None


def create_vec(conn: Any, table: str, dimension: int) -> None:
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
        f"profile_id TEXT PARTITION KEY, "
        f"embedding float[{int(dimension)}] distance_metric=cosine)"
    )


def _shadow_names(conn: Any, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ? ESCAPE '\\'",
        (table.replace("_", "\\_") + "\\_%",),
    ).fetchall()
    return {str(r[0])[len(table) + 1:] for r in rows}


def rename_vec(conn: Any, old: str, new: str) -> None:
    """Rename a vec0 table WITH its shadow tables. Caller holds the txn."""
    if table_exists(conn, new):
        raise RuntimeError(f"cannot rename {old} to {new}: {new} exists")
    shadows = _shadow_names(conn, old)
    missing = set(VEC_SHADOWS) - shadows
    unknown = {s for s in shadows - set(VEC_SHADOWS) if _VEC_SHADOW_LIKE.match(s)}
    if missing or unknown:
        raise RuntimeError(f"vector table {old}: unexpected shadow tables "
                           f"(missing {sorted(missing)}, unknown {sorted(unknown)})")
    conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
    for suffix in VEC_SHADOWS:
        conn.execute(f"ALTER TABLE {old}_{suffix} RENAME TO {new}_{suffix}")


def drop_vec(conn: Any, table: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {table}")


# -- side tables, trigger, purge --------------------------------------------

def ensure_side_tables(conn: Any) -> None:
    for statement in _SIDE_DDL:
        conn.execute(statement)


def drop_side_tables_if_unused(conn: Any) -> None:
    """The trigger and its three tables go together, only when no space needs them."""
    if table_exists(conn, NEXT_VEC) or table_exists(conn, PREV_VEC):
        return
    conn.execute(f"DROP TRIGGER IF EXISTS {TRIGGER}")
    for table in (NEXT_MAP, PREV_MAP, PURGE):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def purge_pending(conn: Any) -> int:
    """Delete vectors whose fact was erased. Caller holds a write txn."""
    if not table_exists(conn, PURGE):
        return 0
    removed = 0
    for space, table in (("next", NEXT_VEC), ("prev", PREV_VEC)):
        rows = [int(r[0]) for r in conn.execute(
            f"SELECT vec_rowid FROM {PURGE} WHERE space = ?", (space,))]
        if rows and table_exists(conn, table):
            for rowid in rows:
                conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            removed += len(rows)
        conn.execute(f"DELETE FROM {PURGE} WHERE space = ?", (space,))
    return removed


def purge_fact(conn: Any, fact_id: str) -> int:
    """Remove one fact's staged and previous vectors (any order of deletes)."""
    if not table_exists(conn, PURGE):
        return 0
    for space, table in (("next", NEXT_MAP), ("prev", PREV_MAP)):
        conn.execute(
            f"INSERT OR IGNORE INTO {PURGE} (space, vec_rowid) "
            f"SELECT ?, vec_rowid FROM {table} WHERE fact_id = ?", (space, fact_id))
        conn.execute(f"DELETE FROM {table} WHERE fact_id = ?", (fact_id,))
    return purge_pending(conn)


def drop_staging(conn: Any) -> None:
    drop_vec(conn, NEXT_VEC)
    if table_exists(conn, NEXT_MAP):
        conn.execute(f"DELETE FROM {NEXT_MAP}")
    if table_exists(conn, PURGE):
        conn.execute(f"DELETE FROM {PURGE} WHERE space = 'next'")
    drop_side_tables_if_unused(conn)


def drop_previous(conn: Any) -> None:
    drop_vec(conn, PREV_VEC)
    if table_exists(conn, PREV_MAP):
        conn.execute(f"DELETE FROM {PREV_MAP}")
    if table_exists(conn, PURGE):
        conn.execute(f"DELETE FROM {PURGE} WHERE space = 'prev'")
    drop_side_tables_if_unused(conn)


def has_previous(conn: Any) -> bool:
    return table_exists(conn, PREV_VEC)


__all__ = [
    "ACTIVE_STATES", "JOBS", "LIVE_VEC", "NEXT_MAP", "NEXT_VEC", "PREV_MAP",
    "PREV_VEC", "PURGE", "SPACE", "TERMINAL_STATES", "TRIGGER", "VEC_SHADOWS",
    "config_from_public", "connect", "create_vec", "drop_previous", "drop_staging",
    "drop_side_tables_if_unused", "drop_vec", "ensure_control_tables",
    "ensure_side_tables", "has_previous", "public_config", "purge_fact",
    "purge_pending", "read_space", "rename_vec", "same_space", "signature_of",
    "table_exists", "vec_dimension", "write_space",
]
