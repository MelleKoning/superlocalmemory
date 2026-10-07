# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The pieces a re-index job is made of: embed, stage, copy, settle.

Embedding happens with NO lock held and yields to interactive recall; each
batch's write is one short ``BEGIN IMMEDIATE`` transaction under the process
write lock, and its hold time is recorded so it can be measured, not assumed.
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

from superlocalmemory.storage import embedding_spaces as sp
from superlocalmemory.storage.embedding_space_swap import (
    StagedRow,
    content_hash,
    fisher_blobs,
    write_batch,
)

logger = logging.getLogger(__name__)

BATCH = max(1, int(os.environ.get("SLM_REINDEX_BATCH", "32")))
PAUSE_S = max(0.0, float(os.environ.get("SLM_REINDEX_PAUSE_S", "0.02")))
#: After each batch the job rests this fraction of the time the batch took, so
#: it never holds more than about half of the daemon's interpreter: a rollback
#: that only copies vectors is pure Python, and back to back it raised recall's
#: median from 1.9 s to 5.3 s on a 22k-fact store copy.
DUTY_REST = max(0.0, float(os.environ.get("SLM_REINDEX_DUTY_REST", "1.0")))
YIELD_MAX_S = 2.0
RETRIES = 3
PROBE_TEXT = "SuperLocalMemory re-index probe"


class StepFailed(RuntimeError):
    """A step that cannot succeed by retrying; the job fails with this message."""


def build_embedder(emb_cfg: Any) -> Any | None:
    from superlocalmemory.core.engine_wiring import init_embedder

    return init_embedder(SimpleNamespace(embedding=emb_cfg))


def close_embedder(embedder: Any) -> None:
    for name in ("shutdown", "unload", "close"):
        method = getattr(embedder, name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:  # freeing a model must not mask the real outcome
                logger.debug("embedder %s failed: %s", name, exc)
            return


def _as_vector(raw: Any, dimension: int, label: str) -> np.ndarray:
    if raw is None:
        raise ValueError(f"the model returned no vector for {label}")
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    if vec.shape[0] != dimension:
        raise StepFailed(
            f"the model produces {vec.shape[0]}-dimensional vectors, not the "
            f"{dimension} this switch was set up for")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"the model returned a vector with NaN/inf for {label}")
    return vec


def probe(embedder: Any, dimension: int) -> None:
    try:
        vectors = list(embedder.embed_batch([PROBE_TEXT]))
    except Exception as exc:
        raise StepFailed(f"the model did not answer a test request: {exc}") from exc
    if len(vectors) != 1:
        raise StepFailed("the model did not answer a test request")
    try:
        _as_vector(vectors[0], dimension, "the test request")
    except ValueError as exc:
        raise StepFailed(str(exc)) from exc


def _yield_to_recall() -> None:
    from superlocalmemory.core.recall_gate import (
        background_work,
        idle_wait_deadline,
        wait_for_foreground_idle,
    )

    with background_work(), idle_wait_deadline(time.monotonic() + YIELD_MAX_S):
        wait_for_foreground_idle()


def embed_texts(embedder: Any, texts: list[str], dimension: int) -> list[np.ndarray]:
    """Embed with retries; StepFailed when it cannot be done."""
    from superlocalmemory.core.recall_gate import background_work

    last: Exception | None = None
    for attempt in range(RETRIES):
        _yield_to_recall()
        try:
            with background_work():
                raw = list(embedder.embed_batch(list(texts)))
            if len(raw) != len(texts):
                raise ValueError(f"{len(raw)} vectors for {len(texts)} memories")
            return [_as_vector(v, dimension, f"memory {i + 1} of the batch")
                    for i, v in enumerate(raw)]
        except StepFailed:
            raise
        except Exception as exc:
            last = exc
            time.sleep(min(4.0, 0.5 * (2 ** attempt)))
    raise StepFailed(f"the model failed {RETRIES} times on one batch: {last}")


def staged_row(fact_id: str, profile_id: str, content: str, vec: np.ndarray) -> StagedRow:
    mean, var = fisher_blobs(vec)
    return StagedRow(fact_id, profile_id, content_hash(content), vec.tobytes(), mean, var)


def copy_previous(conn: Any, facts: list[tuple[int, str, str, str]],
                  dimension: int) -> dict[str, np.ndarray]:
    """Vectors a rollback can take from the previous space: content unchanged."""
    out: dict[str, np.ndarray] = {}
    if not sp.table_exists(conn, sp.PREV_VEC) or sp.vec_dimension(conn, sp.PREV_VEC) != dimension:
        return out
    for _rowid, fact_id, profile_id, content in facts:
        row = conn.execute(f"SELECT vec_rowid, content_hash, profile_id FROM {sp.PREV_MAP} "
                           "WHERE fact_id = ?", (fact_id,)).fetchone()
        if row is None or row[1] != content_hash(content) or str(row[2]) != profile_id:
            continue
        found = conn.execute(f"SELECT embedding FROM {sp.PREV_VEC} WHERE rowid = ?",
                             (int(row[0]),)).fetchone()
        if found is not None:
            out[fact_id] = np.frombuffer(found[0], dtype=np.float32).copy()
    return out


class LockStats:
    """Write-lock hold per staged batch, for the job's own record."""

    def __init__(self, seed: dict | None = None) -> None:
        self.samples: list[float] = list((seed or {}).get("_samples", []))[-500:]

    def record(self, ms: float) -> None:
        self.samples.append(round(ms, 2))
        del self.samples[:-500]

    def summary(self) -> dict:
        if not self.samples:
            return {"batches": 0}
        ordered = sorted(self.samples)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        return {"batches": len(ordered), "lock_hold_ms_p50": ordered[len(ordered) // 2],
                "lock_hold_ms_p95": p95, "lock_hold_ms_max": ordered[-1],
                "_samples": self.samples}


@contextmanager
def write_txn(conn: Any, db_path: Any, stats: LockStats | None = None) -> Iterator[Any]:
    """BEGIN IMMEDIATE under the process write lock; records the hold time."""
    from superlocalmemory.storage.write_lock import get_write_lock

    with get_write_lock(db_path):
        started = time.perf_counter()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            if stats is not None:
                stats.record((time.perf_counter() - started) * 1000)


def stage(conn: Any, db_path: Any, job: dict, embedder: Any, facts: list, dimension: int,
          *, rollback: bool, stats: LockStats, cursor: int | None, catching_up: bool) -> None:
    """Embed (or copy) one batch of facts and stage it in one short transaction."""
    _yield_to_recall()  # copying never reaches the model call that yields
    started = time.perf_counter()
    copied = copy_previous(conn, facts, dimension) if rollback else {}
    todo = [f for f in facts if f[1] not in copied]
    vectors = dict(zip((f[1] for f in todo),
                       embed_texts(embedder, [f[3] for f in todo], dimension))) if todo else {}
    vectors.update(copied)
    rows = [staged_row(fid, pid, content, vectors[fid]) for _r, fid, pid, content in facts]
    with write_txn(conn, db_path, stats):
        write_batch(conn, job, rows, cursor=cursor,
                    done_inc=0 if catching_up else len(facts), copied_inc=len(copied),
                    caught_up_inc=len(facts) if catching_up else 0)
    rest = max(PAUSE_S, (time.perf_counter() - started) * DUTY_REST)
    if rest:
        time.sleep(rest)


def settle(db_path: Any, embedder: Any, model_name: str, dimension: int,
           limit: int = 2000) -> int:
    """After a swap: re-embed any fact whose vector is not in the new space.

    Only a writer of the replaced engine that was mid-write at the swap can
    leave one behind; the count is recorded on the job.
    """
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    conn = sp.connect(db_path)
    try:
        facts = [(str(r[0]), str(r[1]), r[2] or "") for r in conn.execute(
            "SELECT f.fact_id, f.profile_id, f.content FROM atomic_facts f "
            "LEFT JOIN embedding_metadata e ON e.fact_id = f.fact_id "
            "WHERE e.fact_id IS NULL OR e.dimension != ? OR e.model_name != ? LIMIT ?",
            (dimension, model_name, limit))]
        if not facts or embedder is None:
            return 0
        store = VectorStore(db_path, VectorStoreConfig(dimension=dimension, model_name=model_name))
        fixed = 0
        for start in range(0, len(facts), BATCH):
            chunk = facts[start:start + BATCH]
            vectors = embed_texts(embedder, [c[2] for c in chunk], dimension)
            for (fact_id, profile_id, _content), vec in zip(chunk, vectors):
                if not store.upsert(fact_id, profile_id, vec.tolist(), model_name=model_name):
                    continue
                mean, var = fisher_blobs(vec)
                with write_txn(conn, db_path):
                    conn.execute("UPDATE atomic_facts SET embedding = ?, fisher_mean = ?, "
                                 "fisher_variance = ? WHERE fact_id = ?",
                                 (vec.tobytes(), mean, var, fact_id))
                fixed += 1
        return fixed
    finally:
        conn.close()


__all__ = ["BATCH", "LockStats", "StepFailed", "build_embedder", "close_embedder",
           "copy_previous", "embed_texts", "probe", "settle", "stage", "staged_row",
           "write_txn"]
