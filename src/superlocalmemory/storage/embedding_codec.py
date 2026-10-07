"""Encode and decode atomic_facts.embedding values.

All read and write paths for atomic_facts.embedding must go through
``encode_embedding`` and ``decode_embedding``.  Centralising the logic here
means a future format change requires one edit, not one per reader.

Storage format
--------------
New rows: 768 × float32, little-endian, stored as SQLite BLOB (3,072 bytes).
Legacy rows: JSON TEXT produced by json.dumps(list[float]).

The read path accepts both formats so a partial backfill is safe by
construction: callers see ``list[float]`` regardless of storage format.

Error contract
--------------
A value that is neither valid JSON nor a well-formed float32 buffer raises
``ValueError`` with the fact_id in the message.  Returning ``None`` silently
for a malformed value is forbidden: the caller cannot distinguish a legitimate
absent embedding from a data-loss event.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_BYTES",
    "expected_bytes",
    "set_expected_dimension",
    "encode_embedding",
    "decode_embedding",
    "encode_float_vector",
    "decode_float_vector",
]

EMBEDDING_DIM: int = 768
EMBEDDING_BYTES: int = EMBEDDING_DIM * 4  # float32 = 4 bytes

# The width this process's store actually holds. 768 was hard-coded, so on a
# store of any other width (a 384- or 1536-wide model, or after switching to
# one) EVERY read of every fact logged the "truncated write" warning below:
# 852,628 lines in one 40-minute run on a 22k-fact store copy, with recall's
# median at 21 s right after a switch. The engine sets it from the live space
# (core/embedding_live.py); a mismatch is still reported, once per length.
_expected_dim: int = EMBEDDING_DIM
_reported_lengths: set[int] = set()


def set_expected_dimension(dimension: int) -> None:
    """The live space's width (called when an engine binds to its store)."""
    global _expected_dim
    if int(dimension) > 0 and int(dimension) != _expected_dim:
        _expected_dim = int(dimension)
        _reported_lengths.clear()


def expected_bytes() -> int:
    return _expected_dim * 4


def encode_embedding(vec: list[float] | None) -> bytes | None:
    """Serialise a float list to a binary float32 BLOB for SQLite storage.

    Parameters
    ----------
    vec:
        A list of floats, or ``None``.  Production embeddings are always
        768-dimensional; the backfill script asserts the dimension before
        calling this function.

    Returns
    -------
    bytes | None
        Little-endian float32 buffer, or ``None`` if *vec* is ``None``.
    """
    if vec is None:
        return None
    return np.array(vec, dtype=np.float32).tobytes()


def decode_embedding(
    raw: bytes | str | None,
    *,
    fact_id: str = "<unknown>",
) -> list[float] | None:
    """Deserialise an embedding from either TEXT (JSON) or BLOB (binary float32).

    Parameters
    ----------
    raw:
        The raw value from ``atomic_facts.embedding``:
        ``None`` or empty string → absent embedding (returns ``None``).
        ``bytes`` → binary float32 BLOB path.
        ``str`` → legacy JSON TEXT path.
    fact_id:
        Included in any ``ValueError`` message for fast triage.

    Returns
    -------
    list[float] | None
        768-element list of floats, or ``None`` when the embedding is absent.

    Raises
    ------
    ValueError
        For a non-null value that is neither valid JSON nor a well-formed
        float32 buffer.  Never returns ``None`` for a malformed value.
    """
    if raw is None or raw == "":
        return None

    if isinstance(raw, (bytes, bytearray)):
        if len(raw) % 4 != 0 or len(raw) == 0:
            raise ValueError(
                f"Corrupt embedding buffer for fact {fact_id!r}: "
                f"{len(raw)} bytes is not a multiple of 4 (float32)"
            )
        if len(raw) != expected_bytes() and len(raw) not in _reported_lengths:
            _reported_lengths.add(len(raw))
            # A torn write that happens to land on a 4-byte boundary is
            # indistinguishable from a short vector by length alone, and it was
            # accepted silently at debug level: 767 of 768 values still looks
            # like a valid embedding, and every similarity computed from it is
            # quietly wrong. Smaller vectors ARE legitimate in tests, so this is
            # a warning rather than a refusal, but it must be visible.
            logger.warning(
                "embedding for fact %s is %d bytes (%d floats), not the expected "
                "%d (%d floats) — expected only for a test vector; on a real "
                "store this is a truncated write (further vectors of this size "
                "are not reported)",
                fact_id, len(raw), len(raw) // 4, expected_bytes(), _expected_dim,
            )
        return np.frombuffer(raw, dtype=np.float32).tolist()

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Corrupt JSON embedding for fact {fact_id!r}: {exc}"
            ) from exc

    raise ValueError(
        f"Unexpected embedding type {type(raw).__name__!r} for fact {fact_id!r}; "
        f"expected bytes or str"
    )


# ---------------------------------------------------------------------------
# The same treatment for the other float vectors stored on a fact
# ---------------------------------------------------------------------------
#
# A fact carries two more 768-wide vectors besides its embedding: the diagonal
# Fisher mean and variance that the memory dynamics read. They were written as
# JSON text, which costs about 17 KB each against 3 KB for the same numbers as
# float32. Measured on a real 447 MB store: 116.5 MB of Fisher text describing
# 3.6 MB of memories — thirty-two times the size of the content itself, and
# more than a quarter of the whole file.
#
# The read path accepts both forms for the same reason the embedding one does:
# a store converts when a migration reaches it, and everything has to keep
# working in the meantime.


def encode_float_vector(vec: list[float] | None) -> bytes | None:
    """Serialise any float vector to a little-endian float32 BLOB."""
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_float_vector(
    raw: bytes | str | None,
    *,
    field: str = "vector",
    fact_id: str = "<unknown>",
) -> list[float] | None:
    """Read a float vector written as either JSON text or a float32 BLOB.

    Raises rather than returning ``None`` for a malformed value: a caller
    cannot tell a legitimately absent vector from a lost one, and these feed
    the decay dynamics, where a silently empty vector reads as "no evidence"
    instead of "evidence missing".
    """
    if raw is None or raw == "":
        return None

    if isinstance(raw, (bytes, bytearray)):
        if len(raw) == 0 or len(raw) % 4 != 0:
            raise ValueError(
                f"Corrupt {field} buffer for fact {fact_id!r}: {len(raw)} bytes "
                f"is not a multiple of 4 (float32)"
            )
        return np.frombuffer(raw, dtype=np.float32).tolist()

    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Corrupt JSON {field} for fact {fact_id!r}: {exc}"
            ) from exc
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(
                f"{field} for fact {fact_id!r} decoded to "
                f"{type(value).__name__}, expected a list"
            )
        return [float(v) for v in value]

    raise ValueError(
        f"Unexpected {field} type {type(raw).__name__!r} for fact {fact_id!r}; "
        f"expected bytes or str"
    )
