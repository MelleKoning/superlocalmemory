# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Narrowing a recall by project, by the agent that saved a memory, or by what
a memory is about.

``project`` and ``agent`` come from what was recorded when each memory was
saved (the ``project`` and ``agent_id`` an agent passes to remember); they are
compared without regard to case or surrounding spaces. ``about`` is a name - a
person, a project, a tool - looked up the way recall looks up names: exact
name, alias, or a spelling close enough to merge automatically, read-only.

The candidates were already checked for visibility (a shared or global
memory may belong to another profile), so matching does not re-filter by
profile. A facet the caller asks for is a hard filter: only memories that match are
kept, even if that leaves none, because the caller asked for exactly that.
Reads only; never raises (a failure keeps nothing rather than everything).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_MAX_VALUE = 200


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:_MAX_VALUE] if text else None


@dataclass(frozen=True, slots=True)
class Facets:
    project: str | None = None
    agent: str | None = None
    about: str | None = None
    #: 4.1.19 WP8: the DISPLAYED kind (storage.memory_kinds.kind_fields), so a
    #: legacy row matches the kind it is mapped to, not its raw stored value.
    #: Already validated/normalized by the caller (core.kind_query.resolve_kind)
    #: before it ever reaches here — this module only matches, never parses.
    kind: str | None = None

    @classmethod
    def of(cls, project: object = None, agent: object = None, about: object = None,
          kind: object = None) -> "Facets":
        return cls(_clean(project), _clean(agent), _clean(about), _clean(kind))

    @property
    def empty(self) -> bool:
        return (self.project is None and self.agent is None and self.about is None
                and self.kind is None)

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in (("project", self.project), ("agent", self.agent),
                                  ("about", self.about), ("kind", self.kind))
                if v is not None}


def _chunks(items: list[str], size: int = 500) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _by_memory_metadata(db: Any, fact_ids: list[str], profile_id: str,
                        key: str, wanted: str) -> set[str]:
    keep: set[str] = set()
    for chunk in _chunks(fact_ids):
        rows = db.execute(
            "SELECT f.fact_id FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id "
            f"WHERE f.fact_id IN ({','.join('?' * len(chunk))}) "
            f"AND lower(trim(json_extract(m.metadata_json, '$.{key}'))) = ?",
            (*chunk, wanted.lower()),
        )
        keep.update(str(dict(r)["fact_id"]) for r in rows)
    return keep


def _about(db: Any, fact_ids: list[str], profile_id: str, name: str, resolver: Any) -> set[str]:
    entity_ids: set[str] = set()
    lookup = getattr(resolver, "lookup", None)
    if lookup is not None:
        entity_ids.update(lookup([name], profile_id).values())
    else:
        entity = db.get_entity_by_name(name, profile_id)
        if entity is not None:
            entity_ids.add(entity.entity_id)
    if not entity_ids:
        return set()
    keep: set[str] = set()
    for chunk in _chunks(fact_ids):
        rows = db.execute(
            "SELECT fact_id, canonical_entities_json FROM atomic_facts "
            f"WHERE fact_id IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        for row in rows:
            d = dict(row)
            try:
                mentioned = set(json.loads(d.get("canonical_entities_json") or "[]"))
            except (TypeError, ValueError):
                continue
            if mentioned & entity_ids:
                keep.add(str(d["fact_id"]))
    return keep


def _by_kind(db: Any, fact_ids: list[str], wanted: str) -> set[str]:
    from superlocalmemory.storage.memory_kinds import kind_fields

    keep: set[str] = set()
    for chunk in _chunks(fact_ids):
        rows = db.execute(
            "SELECT fact_id, memory_kind, memory_kind_source, memory_kind_confidence, "
            "fact_type FROM atomic_facts "
            f"WHERE fact_id IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        for row in rows:
            d = dict(row)
            if kind_fields(d)["memory_kind"] == wanted:
                keep.add(str(d["fact_id"]))
    return keep


def matching_fact_ids(db: Any, fact_ids: Iterable[str], profile_id: str,
                      facets: Facets, resolver: Any = None) -> set[str]:
    """The subset of ``fact_ids`` that matches every requested facet."""
    remaining = list(dict.fromkeys(str(f) for f in fact_ids))
    if facets.empty or not remaining:
        return set(remaining)
    try:
        if facets.project is not None:
            remaining = [f for f in remaining if f in _by_memory_metadata(
                db, remaining, profile_id, "project", facets.project)]
        if facets.agent is not None and remaining:
            remaining = [f for f in remaining if f in _by_memory_metadata(
                db, remaining, profile_id, "agent_id", facets.agent)]
        if facets.about is not None and remaining:
            remaining = [f for f in remaining if f in _about(
                db, remaining, profile_id, facets.about, resolver)]
        if facets.kind is not None and remaining:
            remaining = [f for f in remaining if f in _by_kind(
                db, remaining, facets.kind)]
    except Exception as exc:  # noqa: BLE001 - a filter that cannot run keeps nothing
        logger.warning("Recall facet filter failed (%s); no results kept", type(exc).__name__)
        return set()
    return set(remaining)


def list_facets(db: Any, profile_id: str, *, limit: int = 50) -> dict[str, list[dict]]:
    """The projects and agents this profile's memories were saved under, with counts."""
    out: dict[str, list[dict]] = {"projects": [], "agents": []}
    for key, label in (("project", "projects"), ("agent_id", "agents")):
        try:
            rows = db.execute(
                f"SELECT lower(trim(json_extract(metadata_json, '$.{key}'))) AS v, COUNT(*) AS n "
                "FROM memories WHERE profile_id = ? GROUP BY v HAVING v IS NOT NULL AND v != '' "
                "ORDER BY n DESC, v LIMIT ?", (profile_id, int(limit)))
        except Exception:  # noqa: BLE001
            continue
        out[label] = [{"name": str(dict(r)["v"]), "memories": int(dict(r)["n"])} for r in rows]
    return out


__all__ = ["Facets", "list_facets", "matching_fact_ids"]
