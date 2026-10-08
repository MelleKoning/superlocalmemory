# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Vector parity: the two vector indexes must say what the memories say.

A memory keeps its own embedding (``atomic_facts.embedding``). Two derived
indexes are built from it: the sqlite-vec table (``fact_embeddings``) and, on a
store promoted to the scale backends, the Lance projection. Measured on a copy
of a real 23,900-memory store, both had drifted:

* 11 sqlite-vec rows held a vector that was not the memory's own embedding (older
  versions wrote them). Lance and the memory agreed, so the sqlite-vec row was
  the odd one out, and the two searches ranked differently;
* 7 Lance rows belonged to memories that were erased, deleted or withheld.

After the 11 rows were rewritten from the memory's embedding and the 7 Lance
rows removed, both searches agreed on 70 of 70 real queries.

This module owns the two definitions the scan and the repair share, so they can
never disagree about what is drift:

* **stale** — a live, visible memory whose sqlite-vec vector is not its own
  embedding. "Not its own embedding" is a cosine below :data:`STALE_COSINE`, not
  a byte comparison: 15,771 of the 23,900 embeddings are still stored as JSON
  text, and the text round trip alone moves a value by up to 1e-7 (cosine 1 -
  1e-14), while the 11 real drifts measured 0.82 to 0.97. Exactly equal bytes
  are fast-pathed. A different width is not drift but a different space, and is
  only counted (a re-embed fixes it, not a copy).
* **orphan** — a Lance row whose memory is gone, or is withheld or soft-deleted:
  the projection worker keeps those out of Lance, so a row for one is a leftover.

Nothing here ever inserts a vector: only an existing row of a memory recall may
return is rewritten, so an erased, deleted or withheld memory's vector can never
come back through this repair.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from superlocalmemory.storage.embedding_projection import decode_embedding_array

logger = logging.getLogger(__name__)

#: Below this cosine the index vector is not the memory's embedding. Float noise
#: from the JSON text round trip is ~1e-14 away from 1; real drift measured <= 0.97.
STALE_COSINE = 0.9999

#: Ids looked up per query when checking Lance rows against SQLite.
LOOKUP_CHUNK = 500

_ROW_SQL = (
    "SELECT af.fact_id, af.profile_id, af.embedding, fe.embedding, em.vec_rowid "
    "FROM atomic_facts af "
    "JOIN embedding_metadata em ON em.fact_id = af.fact_id AND em.profile_id = af.profile_id "
    "JOIN fact_embeddings fe ON fe.rowid = em.vec_rowid AND fe.profile_id = af.profile_id "
    "WHERE af.embedding IS NOT NULL"
)

OK, STALE, DIMENSION, UNUSABLE = "ok", "stale", "dimension", "unusable"


@dataclass(frozen=True)
class StaleRow:
    fact_id: str
    profile_id: str
    vec_rowid: int


@dataclass(frozen=True)
class IndexScan:
    stale: tuple[StaleRow, ...]
    dimension_mismatch: int
    unusable_source: int


def is_stale(source: np.ndarray, indexed: np.ndarray) -> bool:
    """True when ``indexed`` is not ``source`` (same width, caller checked)."""
    if np.array_equal(source, indexed):
        return False
    if not np.all(np.isfinite(indexed)):
        return True
    a = source.astype(np.float64)
    b = indexed.astype(np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return True
    return float(a @ b) / (na * nb) < STALE_COSINE


def classify(raw_source: Any, indexed_blob: Any, fact_id: str) -> tuple[str, np.ndarray | None]:
    """``(kind, source vector)`` for one memory and the row indexed for it."""
    try:
        source = decode_embedding_array(raw_source, fact_id=fact_id)
    except (ValueError, TypeError):
        return UNUSABLE, None
    if source is None or source.ndim != 1 or source.size == 0:
        return UNUSABLE, None
    if not isinstance(indexed_blob, (bytes, bytearray)) or len(indexed_blob) % 4:
        return UNUSABLE, None
    indexed = np.frombuffer(indexed_blob, dtype=np.float32)
    if indexed.shape != source.shape:
        return DIMENSION, None
    if not np.all(np.isfinite(source)) or not float(np.linalg.norm(source.astype(np.float64))):
        return UNUSABLE, None  # nothing trustworthy to copy from
    return (STALE if is_stale(source, indexed) else OK), source


def _visible(conn: sqlite3.Connection) -> str:
    from superlocalmemory.storage.database import visible_fact_clause_for_connection

    return visible_fact_clause_for_connection(conn, prefix="af")


def scan_index(conn: sqlite3.Connection) -> IndexScan:
    """Every live, visible memory whose sqlite-vec row is not its own embedding.

    ``conn`` must have the vector extension loaded. Read-only.
    """
    stale: list[StaleRow] = []
    dimension = unusable = 0
    for fact_id, profile_id, raw, blob, rowid in conn.execute(_ROW_SQL + _visible(conn)):
        kind, _ = classify(raw, blob, str(fact_id))
        if kind == STALE:
            stale.append(StaleRow(str(fact_id), str(profile_id), int(rowid)))
        elif kind == DIMENSION:
            dimension += 1
        elif kind == UNUSABLE:
            unusable += 1
    return IndexScan(tuple(stale), dimension, unusable)


def rewrite_stale(conn: sqlite3.Connection, rows: Iterable[StaleRow]) -> list[str]:
    """Rewrite each row from its memory's own embedding; the ids rewritten.

    Call inside the caller's write transaction. Every row is checked again here
    (the scan is only a list of names): a memory that was withheld, deleted or
    re-embedded since is skipped, and only an EXISTING row is updated - never an
    insert.
    """
    done: list[str] = []
    visible = _visible(conn)
    for row in rows:
        found = conn.execute(
            _ROW_SQL + " AND af.fact_id = ? AND em.vec_rowid = ?" + visible,
            (row.fact_id, row.vec_rowid)).fetchone()
        if found is None:
            continue
        kind, source = classify(found[2], found[3], row.fact_id)
        if kind != STALE or source is None:
            continue
        blob = np.ascontiguousarray(source, dtype="<f4").tobytes()
        cur = conn.execute("UPDATE fact_embeddings SET embedding = ? WHERE rowid = ?",
                           (blob, row.vec_rowid))
        if cur.rowcount == 1:
            done.append(row.fact_id)
    return done


def reindex_in_progress(conn: sqlite3.Connection) -> bool:
    """A model switch is building a new space: leave the live one to it."""
    from superlocalmemory.storage import embedding_spaces as sp

    if not sp.table_exists(conn, sp.JOBS):
        return False
    marks = ",".join("?" * len(sp.ACTIVE_STATES))
    return conn.execute(f"SELECT 1 FROM {sp.JOBS} WHERE state IN ({marks}) LIMIT 1",  # noqa: S608
                        sp.ACTIVE_STATES).fetchone() is not None


# -- the Lance side -----------------------------------------------------------

NOT_ACTIVE, ACTIVE, NOT_INSTALLED, UNREADABLE = ("not_active", "active", "not_installed",
                                                 "unreadable")


def lance_state(db_path: str | Path) -> str:
    """Whether this store serves vector search from Lance (read-only; no import).

    Active means all of: the config says the store was promoted to a Lance
    vector backend (the rule ``BackendOrchestrator`` applies at start-up), the
    last start-up found it ``active``, and its table is on disk. Anything less
    and nothing here touches Lance. ``installed`` is judged last, by
    ``find_spec`` (importing lancedb starts a native runtime).
    """
    root = Path(db_path).parent
    try:
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return NOT_ACTIVE
    if not isinstance(cfg, dict) or str(cfg.get("scale_engine_state", "")).lower() != "promoted":
        return NOT_ACTIVE
    if str(cfg.get("vector_backend", "auto") or "auto") not in ("auto", "lancedb"):
        return NOT_ACTIVE
    if not (root / "lance" / "embeddings.lance").exists():
        return NOT_ACTIVE
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        try:
            row = conn.execute("SELECT status FROM backend_status WHERE backend_name = "
                               "'lancedb'").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return NOT_ACTIVE
    if row is None or str(row[0]) != "active":
        return NOT_ACTIVE
    try:
        present = importlib.util.find_spec("lancedb") is not None
    except Exception:  # a broken install: treat as absent
        present = False
    return ACTIVE if present else NOT_INSTALLED


def orphan_ids(conn: sqlite3.Connection, fact_ids: Iterable[str]) -> list[str]:
    """The ids with no live, visible memory behind them (gone, withheld, soft-deleted).

    Asked one profile at a time: ``idx_facts_visibility`` (storage/visibility_index)
    is led by ``profile_id`` and holds every column the visibility rule reads, so
    each lookup is answered from the index. Without the profile the planner reads
    the rows instead, and ``archive_status`` / ``quarantined`` sit about 40 KB into
    each one: 43 s for 23,500 ids on a copy of a real store, against the index's
    seconds.
    """
    pending = list(dict.fromkeys(str(i) for i in fact_ids))
    visible = _visible(conn)
    profiles = [str(r[0]) for r in conn.execute("SELECT DISTINCT profile_id FROM atomic_facts")]
    for profile_id in profiles:
        if not pending:
            break
        shown: set[str] = set()
        for start in range(0, len(pending), LOOKUP_CHUNK):
            chunk = pending[start:start + LOOKUP_CHUNK]
            marks = ",".join("?" * len(chunk))
            shown.update(str(r[0]) for r in conn.execute(
                f"SELECT af.fact_id FROM atomic_facts af WHERE af.profile_id = ? "  # noqa: S608
                f"AND af.fact_id IN ({marks}){visible}", (profile_id, *chunk)))
        pending = [i for i in pending if i not in shown]
    return pending


def _lance_ids(db_path: str | Path, lance: Any) -> tuple[str, list[str] | None]:
    if lance is False:  # the caller runs no projection: there is nothing to look at
        return NOT_ACTIVE, None
    if lance is not None:
        try:
            return ACTIVE, list(lance.fact_ids())
        except Exception as exc:  # noqa: BLE001 - a scan must report, not raise
            logger.warning("vector parity: Lance rows unreadable: %s", type(exc).__name__)
            return UNREADABLE, None
    state = lance_state(db_path)
    if state != ACTIVE:
        return state, None
    try:
        from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

        ids = LanceDBVectorBackend.read_fact_ids(Path(db_path).parent / "lance")
    except Exception as exc:  # noqa: BLE001
        logger.warning("vector parity: Lance rows unreadable: %s", type(exc).__name__)
        return UNREADABLE, None
    return (ACTIVE, ids) if ids is not None else (NOT_ACTIVE, None)


def lance_census(conn: sqlite3.Connection, db_path: str | Path, lance: Any = None) -> dict[str, Any]:
    state, ids = _lance_ids(db_path, lance)
    if ids is None:
        return {"state": state, "rows": None, "orphans": None}
    return {"state": state, "rows": len(ids), "orphans": len(orphan_ids(conn, ids))}


def census(conn: sqlite3.Connection, db_path: str | Path, lance: Any = None) -> dict[str, Any]:
    """Counts only: stale sqlite-vec rows and orphan Lance rows.

    ``stale_vectors`` is None when the vector extension cannot be loaded here
    (not checked), like ``unreachable_vectors``. ``lance`` is the running
    projection when the caller has one, ``False`` when the caller runs none, and
    None to find it, read-only, on disk.
    """
    from superlocalmemory.storage.vector_residue import vec_connection

    out: dict[str, Any] = {"stale_vectors": None, "dimension_mismatch": None,
                           "unusable_source": None}
    with vec_connection(db_path) as vconn:
        if vconn is not None:
            found = scan_index(vconn)
            out.update(stale_vectors=len(found.stale), dimension_mismatch=found.dimension_mismatch,
                       unusable_source=found.unusable_source)
    out["lance"] = lance_census(conn, db_path, lance)
    return out


__all__ = ["IndexScan", "STALE_COSINE", "StaleRow", "census", "classify", "is_stale",
           "lance_census", "lance_state", "orphan_ids", "reindex_in_progress", "rewrite_stale",
           "scan_index"]
