# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A per-profile counter that moves whenever that profile's graph changes.

WHY
---
Spreading activation caches its activations for an hour, keyed by the question
and the seed set. The activations are a function of the seeds AND the links the
walk follows, so a link added, removed or re-weighted after a question was
asked -- one that changes the walk without changing the seeds -- stayed hidden
from that question for up to the full hour. The counter is the missing half of
the key: ``graph_generation.generation`` for a profile changes on every write to
that profile's ``graph_edges`` or ``association_edges`` rows, so an entry written
before the change is never read after it.

HOW
---
Triggers, not writer-side calls. Edges are written from many places (the store,
the linker, the pruner, restore, migrations) and also removed by foreign-key
cascades when a memory or profile is deleted; a trigger sees every one of those,
from every process, including writers that do not exist yet. A trigger costs a
primary-key upsert per changed edge row.

Memory (node) writes do not bump the counter, on purpose. A new or edited memory
changes the activations only through the seeds (already in the key) or through
its links (which bump the counter). A deleted memory takes its association links
with it by cascade (which bumps it), and a cache hit is re-filtered against the
store before it is returned. Bumping on every ``atomic_facts`` write would also
bump on the access counters recall itself updates, and the cache would never hit.

The guard ``profile_id IN (SELECT profile_id FROM profiles)`` lets a profile be
deleted: its edges cascade away while its own row is already gone, and an upsert
for it would then fail the foreign key and abort the delete.

Deleting a profile empties ``activation_cache``. A cross-scope entry of another
profile may have walked the deleted profile's global or shared links, and a
profile re-created under the same id restarts its counter -- the only way a
counter value could repeat. Emptying the cache on delete closes that.

Applied by ``schema.create_all_tables`` on every engine start, like
``projection_outbox``, so no migration (and no pre-upgrade copy of memory.db) is
needed, and freshness never depends on a migration having run. A store without
the table cannot prove freshness, so :func:`graph_version` raises and the
caller does not use the cache at all.
"""

from __future__ import annotations

import hashlib
from typing import Any

TABLE = "graph_generation"

#: The edge tables whose rows the walk reads.
EDGE_TABLES: tuple[str, ...] = ("graph_edges", "association_edges")

_BUMP = """
    INSERT INTO graph_generation (profile_id, generation)
    SELECT p, 1 FROM ({source})
    WHERE p IN (SELECT profile_id FROM profiles)
    ON CONFLICT (profile_id) DO UPDATE SET generation = generation + 1;"""

_SOURCES = {
    "insert": "SELECT NEW.profile_id AS p",
    "delete": "SELECT OLD.profile_id AS p",
    "update": "SELECT NEW.profile_id AS p UNION SELECT OLD.profile_id",
}


def trigger_name(table: str, event: str) -> str:
    return f"trg_{table}_graph_generation_{event}"


#: Deleting a profile empties the activation cache (see module docstring).
PROFILE_TRIGGER = "trg_profiles_activation_cache_reset"


def _edge_triggers() -> str:
    parts = []
    for table in EDGE_TABLES:
        for event, source in _SOURCES.items():
            parts.append(
                f"CREATE TRIGGER IF NOT EXISTS {trigger_name(table, event)}\n"
                f"AFTER {event.upper()} ON {table}\nBEGIN"
                f"{_BUMP.format(source=source)}\nEND;"
            )
    return "\n".join(parts)


DDL = f"""
CREATE TABLE IF NOT EXISTS graph_generation (
    profile_id  TEXT PRIMARY KEY,
    generation  INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id) ON DELETE CASCADE
);
{_edge_triggers()}
CREATE TRIGGER IF NOT EXISTS {PROFILE_TRIGGER}
AFTER DELETE ON profiles
BEGIN
    DELETE FROM activation_cache;
END;
"""


def trigger_names() -> tuple[str, ...]:
    """Every trigger :data:`DDL` creates, for drop and verification."""
    return tuple(
        trigger_name(table, event)
        for table in EDGE_TABLES for event in _SOURCES
    ) + (PROFILE_TRIGGER,)


def graph_version(db: Any, profile_id: str, *, cross_scope: bool) -> str:
    """A cache-key fragment that changes whenever the walkable graph changes.

    A personal read walks only the profile's own links, so its own counter is
    enough. A cross-scope read can walk any profile's global or shared links,
    so it is keyed by every profile's counter.

    Raises whatever the read raises (for example a store without the table):
    the caller must then neither read nor write the cache.
    """
    if not cross_scope:
        rows = db.execute(
            "SELECT generation FROM graph_generation WHERE profile_id = ?",
            (profile_id,),
        )
        generation = int(dict(rows[0])["generation"]) if rows else 0
        return f"|graph={generation}"
    rows = db.execute(
        "SELECT profile_id, generation FROM graph_generation "
        "ORDER BY profile_id",
        (),
    )
    material = "\n".join(
        f"{dict(r)['profile_id']}={int(dict(r)['generation'])}" for r in rows
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"|graph=all:{digest}"
