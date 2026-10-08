# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which stored meaning vectors are missing from the vector index, found cheaply.

WHY
---
At every daemon start the self-heal step looks for facts whose embedding exists
but that the sqlite-vec index cannot reach. It used to answer that by loading
every fact of every profile (``get_all_facts``: ``SELECT *`` plus decoding each
768-float embedding and both Fisher vectors) and only then comparing ids.
Measured on a copy of a 22,175-fact store, during the first minute after start:
the read alone took 4.8-18 s, the decoding held the interpreter lock for
seconds while recalls waited behind it, and the daemon grew by about 1.2 GB.
It did this on every start, including the ones where nothing was missing.

WHAT THIS DOES
--------------
The same answer from ids: the visible facts of the profile (read from the
visibility index, no row bodies), minus the facts the vector index already
holds. Only the remainder -- normally a handful, or none -- has its embedding
read and decoded. The result is the same list the old scan produced: every
visible fact of the profile whose embedding has the configured dimension and
has no vector the index can reach.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: ids per ``IN (...)`` read; well inside SQLITE_MAX_VARIABLE_NUMBER.
_CHUNK = 800


def profile_ids(db: Any) -> list[str]:
    """Every profile of the store; ``default`` when the table cannot be read."""
    try:
        ids = [str(dict(r)["profile_id"]) for r in
               db.execute("SELECT profile_id FROM profiles ORDER BY profile_id")]
    except Exception as exc:  # noqa: BLE001 -- the repair still covers default
        logger.warning("vector repair: profiles unreadable (%s); default only", type(exc).__name__)
        ids = []
    return ids or ["default"]


def visible_fact_ids(db: Any, profile_id: str) -> list[str]:
    """Every fact id of ``profile_id`` a caller may see, without reading row bodies."""
    rows = db.execute(
        "SELECT fact_id FROM atomic_facts WHERE profile_id = ?"
        f"{db.visible_fact_clause()} ORDER BY fact_id",
        (profile_id,),
    )
    return [str(dict(r)["fact_id"]) for r in rows]


def missing_vectors(db: Any, store: Any, profile_id: str,
                    dimension: int) -> list[tuple[str, str, list[float]]]:
    """``(fact_id, profile_id, embedding)`` for each visible fact the index lacks."""
    from superlocalmemory.storage.embedding_projection import fetch_fact_embeddings_by_ids

    indexed = store.indexed_fact_ids(profile_id)
    candidates = [fid for fid in visible_fact_ids(db, profile_id) if fid not in indexed]
    missing: list[tuple[str, str, list[float]]] = []
    for start in range(0, len(candidates), _CHUNK):
        batch = candidates[start:start + _CHUNK]
        for fact_id, vector in fetch_fact_embeddings_by_ids(db, batch, profile_id):
            if len(vector) == dimension:
                missing.append((str(fact_id), profile_id, [float(x) for x in vector]))
    missing.sort(key=lambda item: item[0])
    return missing


#: Longest one batch waits for in-flight recalls: steady recall traffic slows the
#: repair but can never stop it (the embedding repair uses the same bound).
RECALL_YIELD_MAX_SECONDS = 30.0


def upsert_missing(db: Any, store: Any, profile_id: str,
                   missing: list[tuple[str, str, list[float]]], *,
                   batch: int, pause: float) -> int:
    """Index ``missing`` in bounded batches; returns how many were written.

    Each batch first waits while a person's recall runs, then takes the store's
    write lock once, and a short pause after it lets a waiting user write in.
    """
    import time

    from superlocalmemory.core import recall_gate

    written = 0
    for start in range(0, len(missing), max(1, batch)):
        recall_gate.yield_to_recalls(RECALL_YIELD_MAX_SECONDS)
        with db._lock:
            for fact_id, owner, embedding in missing[start:start + max(1, batch)]:
                try:
                    if store.upsert(fact_id, owner, embedding):
                        written += 1
                except Exception as exc:  # noqa: BLE001 -- one bad vector never stops the repair
                    logger.warning("VS backfill[%s]: upsert failed for %s: %s",
                                   profile_id, str(fact_id)[:16], exc)
        if pause > 0:
            time.sleep(pause)
    return written


__all__ = ["RECALL_YIELD_MAX_SECONDS", "missing_vectors", "profile_ids",
           "upsert_missing", "visible_fact_ids"]
