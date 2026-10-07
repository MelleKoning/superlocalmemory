# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall's read-only entity name lookup, without scanning the tables each time.

``EntityResolver.lookup`` (recall only; never saving) found each name in a
question by an exact match, then an alias, then the closest spelling, and each
step scanned a whole table: ``LOWER(canonical_name) = LOWER(?)`` has no index,
and the spelling match read every name and alias and scored them all. On a
22k-fact store that was 46 ms of a recall, every recall.

This keeps those rows in memory per profile while nothing has been committed
to the store (storage/store_signature) and answers the same way:

* exact name and alias keys are lowered the way SQLite's ``LOWER`` does (ASCII
  only), so a name matches here exactly when it matched there;
* a key held by more than one entity is answered by the original query, so
  which row wins never depends on a query plan;
* the spelling match walks the rows in the order of the same queries it ran
  before, with the same strict comparison, so ties resolve the same way; its
  answer per name is kept while the rows are.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from superlocalmemory.storage import store_signature

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")

#: The original queries the spelling match ran; their row order is reused.
_NAMES_SQL = "SELECT entity_id, canonical_name FROM canonical_entities WHERE profile_id = ?"
_ALIASES_SQL = ("SELECT ea.entity_id, ea.alias FROM entity_aliases ea "
                "JOIN canonical_entities ce ON ce.entity_id = ea.entity_id "
                "WHERE ce.profile_id = ?")

AMBIGUOUS = object()


def sql_lower(text: str) -> str:
    """SQLite's built-in LOWER(): ASCII letters only."""
    return text.translate(_ASCII_LOWER)


def _keyed(rows: list[tuple[str, str]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for entity_id, text in rows:
        key = sql_lower(text)
        prev = out.get(key)
        if prev is None:
            out[key] = entity_id
        elif prev is not AMBIGUOUS and prev != entity_id:
            out[key] = AMBIGUOUS
    return out


@dataclass
class _Index:
    names: list[tuple[str, str]]
    aliases: list[tuple[str, str]]
    by_name: dict[str, object]
    by_alias: dict[str, object]
    fuzzy: dict[str, tuple[str | None, float]] = field(default_factory=dict)


class EntityLookupIndex:
    """Per-profile name rows, valid while the store signature is unchanged."""

    def __init__(self, db: Any) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._parts: dict[str, tuple[Any, _Index]] = {}

    def get(self, profile_id: str) -> _Index | None:
        sig = store_signature.of_db(self._db)
        if sig is None:
            return None
        with self._lock:
            hit = self._parts.get(profile_id)
            if hit is not None and hit[0] == sig:
                return hit[1]
        names = [(str(dict(r)["entity_id"]), str(dict(r)["canonical_name"] or ""))
                 for r in self._db.execute(_NAMES_SQL, (profile_id,))]
        aliases = [(str(dict(r)["entity_id"]), str(dict(r)["alias"] or ""))
                   for r in self._db.execute(_ALIASES_SQL, (profile_id,))]
        index = _Index(names, aliases, _keyed(names), _keyed(aliases))
        with self._lock:
            self._parts[profile_id] = (sig, index)
            while len(self._parts) > 8:  # a few profiles at most
                self._parts.pop(next(iter(self._parts)))
        return index


def fuzzy_best(index: _Index, name: str,
               score: Callable[[str, str], float]) -> tuple[str | None, float]:
    """The original spelling match over the kept rows (same order, same ``>``)."""
    name_lower = name.lower()
    cached = index.fuzzy.get(name_lower)
    if cached is not None:
        return cached
    best_id: str | None = None
    best = 0.0
    for rows in (index.names, index.aliases):
        for entity_id, text in rows:
            s = score(name_lower, text.lower())
            if s > best:
                best, best_id = s, entity_id
    index.fuzzy[name_lower] = (best_id, best)
    return best_id, best


__all__ = ["AMBIGUOUS", "EntityLookupIndex", "fuzzy_best", "sql_lower"]
