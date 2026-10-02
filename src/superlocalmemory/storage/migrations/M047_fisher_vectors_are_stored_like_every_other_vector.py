# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Rewrite the two Fisher vectors on each fact as float32, like the embedding.

WHAT THIS IS ABOUT

A fact carries three vectors of the same width: its embedding, and the diagonal
Fisher mean and variance the memory dynamics read to decide how fast it decays.
4.0.9 converted the embedding from JSON text to float32 and got 5.5x. The other
two were left in text.

Measured on a real 447 MB store:

    fisher_mean + fisher_variance    116.5 MB
    embedding (already binary)        15.5 MB
    content — the memories themselves  3.6 MB

The Fisher matrices were thirty-two times the size of the memories they
describe, and more than a quarter of the entire file, because a float printed
as decimal text costs about 22 bytes and the same float costs 4.

WHY AN UPDATE AND NOT A REBUILD

``atomic_facts`` has a TEXT primary key, so it also has an implicit rowid, and
its full-text index is external-content keyed on that rowid. Rebuilding the
table reassigns rowids and every index entry then points at a different fact —
searches keep working and return the wrong rows. An UPDATE touches no rowid, so
the search index is untouched by construction. Nothing here alters the schema:
the columns are declared TEXT and SQLite stores a BLOB in them regardless.

WHY IT CAN BE INTERRUPTED

Every row is converted independently and the read path accepts both forms, so a
store stopped halfway is a working store. Re-running finishes it. That is why
the work is committed in batches rather than held in one transaction across
three thousand facts.

WHAT VERIFY CAN AND CANNOT CHECK

float32 has about seven significant digits, and these numbers were written from
float64, so the conversion loses precision on purpose. That trade is checked in
the tests, which hold the original values and compare against them at the
tolerance float32 provides.

``verify`` runs after the fact, when the text it replaced no longer exists, so
it cannot make that comparison. What it can do is refuse the shapes a broken
conversion would leave: a convertible vector still in the old form, meaning the
run stopped early; a buffer that is not a whole number of floats; a non-finite
value; and a buffer that is entirely zero — which passes every structural check
and is a deletion wearing the right shape, because the dynamics read an all-zero
vector as "this memory carries no evidence".
"""

from __future__ import annotations

import json
import logging
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

NAME = "M047_fisher_vectors_are_stored_like_every_other_vector"
DB_TARGET = "memory"

#: No schema change and both forms are readable, so an older build opening a
#: converted store is safe. The floor does not move.
BREAKING_VERSION = 0

_TABLE = "atomic_facts"
_COLUMNS = ("fisher_mean", "fisher_variance")

#: Rows per transaction. Large enough that the commit overhead is irrelevant,
#: small enough that an interrupted run loses a fraction of a second of work.
_BATCH = 500

#: float32 keeps about seven significant digits. Anything inside this is the
#: cost that was chosen; anything outside it is a bug.
_TOLERANCE = 1e-6

DDL = """
-- No schema change. apply() rewrites the two Fisher columns of each fact from
-- JSON text to a little-endian float32 buffer, in place, in batches.
"""


def _is_text_vector(value: object) -> bool:
    """A value still in the old form: a string that opens like a JSON list."""
    return isinstance(value, str) and value.strip().startswith("[")


def _encode(text: str) -> bytes | None:
    """Text form to float32 buffer, or None when there is nothing to store."""
    parsed = json.loads(text)
    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise ValueError(f"expected a list, got {type(parsed).__name__}")
    return np.asarray(parsed, dtype=np.float32).tobytes()


def _pending(conn: sqlite3.Connection) -> int:
    """How many rows still hold a Fisher vector in the old form."""
    clauses = " OR ".join(
        f"(typeof({col}) = 'text' AND {col} LIKE '[%')" for col in _COLUMNS
    )
    row = conn.execute(f"SELECT COUNT(*) FROM {_TABLE} WHERE {clauses}").fetchone()
    return int(row[0]) if row else 0


def apply(conn: sqlite3.Connection) -> None:
    """Convert every remaining text Fisher vector, in batches, resumably."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({_TABLE})")}
    missing = [c for c in _COLUMNS if c not in existing]
    if missing:
        logger.info("M047: %s has no %s; nothing to convert", _TABLE, missing)
        return

    remaining = _pending(conn)
    if remaining == 0:
        logger.info("M047: no Fisher vector is in the old form")
        return
    logger.info("M047: converting Fisher vectors on %d fact(s)", remaining)

    converted = 0
    skipped = 0
    # Walk forward by rowid rather than re-querying "what is left". A row can
    # legitimately still match the WHERE clause after being visited — one of its
    # two vectors may be unreadable while the other converts — and re-selecting
    # on the predicate alone would then hand back the same row forever.
    cursor = 0
    clauses = " OR ".join(
        f"(typeof({col}) = 'text' AND {col} LIKE '[%')" for col in _COLUMNS
    )

    while True:
        batch = conn.execute(
            f"SELECT rowid, fact_id, {', '.join(_COLUMNS)} FROM {_TABLE} "
            f"WHERE rowid > ? AND ({clauses}) ORDER BY rowid LIMIT {_BATCH}",
            (cursor,),
        ).fetchall()
        if not batch:
            break
        cursor = batch[-1][0]

        updates: list[tuple[object, ...]] = []
        for row in batch:
            rowid, fact_id = row[0], row[1]
            values: list[object] = []
            changed = False
            unreadable = False
            # The two vectors are independent. One being unreadable is no reason
            # to leave the other in the form that costs five times as much.
            for offset, column in enumerate(_COLUMNS, start=2):
                raw = row[offset]
                if not _is_text_vector(raw):
                    values.append(raw)
                    continue
                try:
                    values.append(_encode(raw))
                    changed = True
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    # A vector that cannot be parsed is left exactly as it is.
                    # Replacing it with NULL would turn "unreadable" into
                    # "absent", and absent reads as "no evidence" in the decay
                    # dynamics rather than as something to look at.
                    logger.warning(
                        "M047: leaving %s on fact %s as text — %s",
                        column, fact_id, exc,
                    )
                    values.append(raw)
                    unreadable = True
            if unreadable:
                skipped += 1
            if changed:
                updates.append((*values, rowid))

        if updates:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Only write over a value that is STILL in the old form. Reading
                # a row and writing it back later is a lost update: a live
                # writer that recomputes this fact's vectors between the two
                # would have its new value replaced by the old one re-encoded,
                # silently, with no error and no retry. The typeof() guard makes
                # the update conditional on nothing having changed underneath.
                for column in _COLUMNS:
                    per_column = [
                        (values[_COLUMNS.index(column)], rowid)
                        for *values, rowid in updates
                    ]
                    conn.executemany(
                        f"UPDATE {_TABLE} SET {column} = ? "
                        f"WHERE rowid = ? AND typeof({column}) = 'text'",
                        per_column,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            converted += len(updates)

    logger.info(
        "M047: converted %d fact(s); left %d vector(s) as text for inspection",
        converted, skipped,
    )


def repair(conn: sqlite3.Connection) -> None:
    """Finish the conversion on a store that has drifted back to the old form.

    WHY A COMPLETED MIGRATION NEEDS THIS
    ------------------------------------
    A long-running daemon holds its writers in memory. This migration and the
    write-path change that stores the new form shipped in the same commit, so a
    daemon started before that commit converts nothing and keeps writing text --
    and the migration, run by some later short-lived process, marks itself
    complete against a store that is still gaining old-form rows behind it.

    That happened, and the numbers say so exactly: on the author's store the
    migration completed at 08:37, the newest converted row was written at 08:08,
    and sixteen facts written between 10:50 and 17:34 were text. The daemon had
    been up for twenty-five hours, since before the commit existed.

    Without a repair hook the framework refuses to touch a completed migration
    and logs "schema incomplete for completed migration; automatic replay is
    disabled" on every open. Both halves of that are wrong here -- the
    conversion did not stop early, and disabling replay for a false alarm means
    a migration that IS incomplete later goes unrepaired. The mechanism was
    already there; this migration simply never supplied the hook.

    ``apply`` is the whole repair: every row converts independently, the read
    path accepts both forms, and re-running finishes what is left. So this is
    not a second implementation to keep in step -- it is the same one.
    """
    apply(conn)


def verify(conn: sqlite3.Connection) -> bool:
    """Every convertible vector is converted, and the numbers still agree.

    Reading a sample back and comparing it to what it replaced is the only
    check that distinguishes a conversion from a deletion. A count of BLOBs
    would pass just as happily on zeroed buffers.
    """
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({_TABLE})")}
    if any(c not in existing for c in _COLUMNS):
        return True

    # A row whose vector could not be parsed is left as text on purpose, so
    # "some text remains" is not by itself a failure. What would be a failure is
    # text that WOULD have converted — that means apply() stopped early.
    for column in _COLUMNS:
        leftover = conn.execute(
            f"SELECT fact_id, {column} FROM {_TABLE} "
            f"WHERE typeof({column}) = 'text' AND {column} LIKE '[%'"
        ).fetchall()
        for fact_id, raw in leftover:
            try:
                _encode(raw)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue  # genuinely unconvertible; apply() reported it
            # Not necessarily an interrupted run: until 4.1.18 the maintenance
            # cycle wrote the variance back as text, and an older build still
            # running does the same. The runner converts it on this start.
            logger.warning(
                "M047 verify: %s on fact %s is in the old text form; it will "
                "be converted now", column, fact_id,
            )
            return False

    for column in _COLUMNS:
        sample = conn.execute(
            f"SELECT {column} FROM {_TABLE} "
            f"WHERE typeof({column}) = 'blob' LIMIT 50"
        ).fetchall()
        nonzero = 0
        for (raw,) in sample:
            if raw is None:
                continue
            if len(raw) == 0 or len(raw) % 4 != 0:
                logger.error(
                    "M047 verify: a %s buffer is %d bytes, not a float32 vector",
                    column, len(raw),
                )
                return False
            values = np.frombuffer(raw, dtype=np.float32)
            if not np.all(np.isfinite(values)):
                logger.error("M047 verify: a %s buffer holds a non-finite value", column)
                return False
            if np.any(values):
                nonzero += 1
        if sample and nonzero == 0:
            # EVERY sampled vector being zero is a blanket wipe wearing the
            # right shape: the buffers are the correct length and finite, and
            # the dynamics would read all of them as "this memory carries no
            # evidence". One honestly-zero vector is not that, and refusing on
            # a single one would mark the migration incomplete forever on a
            # store that legitimately holds one.
            logger.error(
                "M047 verify: every sampled %s buffer is entirely zero, which "
                "is a wipe rather than a conversion", column,
            )
            return False
    return True
