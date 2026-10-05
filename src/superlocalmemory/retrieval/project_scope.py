# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Using the project a recall is about (GitHub #150).

Two ways to say it, both opt-in per recall:

* ``prefer_project`` - a soft preference. Memories saved under that project
  rank above others of *similar* relevance; nothing is removed, and memories
  saved with no project (most of any store written before 4.1.21) stay
  exactly as findable as before. ``session_init`` passes the session's project
  this way.
* ``project`` - a filter. Only memories saved under that project are kept.
  When none of the memories found for the question were saved under it - a
  project with no tagged memories, a misspelt name - the filter is not applied,
  the unfiltered results are returned, and the response says so
  (``project_scope.filter.applied: false`` with a plain-English ``note``). A
  filter that silently returns nothing is indistinguishable from "this store
  knows nothing", which is the wrong answer.

Both compare projects with ``core.project_identity.project_key`` and read the
project each memory was saved with (``memories.metadata_json -> project``).
The candidates were already admitted for this profile and scope by retrieval,
so this module only narrows or reorders them - it never reaches across
profiles and never adds a candidate retrieval did not find.

Deterministic: the same candidates and the same project always give the same
order (ties keep the incoming order, which retrieval already breaks by id).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from superlocalmemory.core.project_identity import project_key

logger = logging.getLogger(__name__)

#: How much a same-project memory's ranking score is lifted: x(1 + BOOST).
#:
#: Why 0.25. The bound is the point: a same-project memory passes another
#: memory only when its own score is already at least 1/1.25 = 80% of that
#: memory's. A weakly relevant project memory can never jump a strongly
#: relevant one - at two thirds of the other's score it stays below it.
#: Measured on the ranking fixture (tests/test_retrieval/
#: test_project_scope_ranking.py): the on-topic project memories that
#: cross-project trivia outranked sat at 0.84-0.98 of the trivia's ranking
#: score, so 0.25 puts every one of them first; one on-topic memory at
#: 0.61-0.65 of the trivia stays below it, which is the bound doing its job. Half the
#: kind-aware ceiling (retrieval.kind_aware.MAX_BOOST). A constant, not a
#: setting, so two installations rank the same question alike.
BOOST = 0.25

_CHUNK = 500


def lift(score: float) -> float:
    """``score`` lifted by BOOST, relative to its magnitude so a negative
    score is lifted toward zero, never sunk."""
    return float(score) + BOOST * abs(float(score))


def _chunks(items: Sequence[str]) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), _CHUNK):
        yield items[start:start + _CHUNK]


def stored_projects(db: Any, fact_ids: Iterable[str]) -> dict[str, str]:
    """``{fact_id: project}`` as each memory was saved, for those saved with
    one. Raises what the store raises (callers decide what a failure means)."""
    ids = list(dict.fromkeys(str(f) for f in fact_ids))
    found: dict[str, str] = {}
    for chunk in _chunks(ids):
        rows = db.execute(
            "SELECT f.fact_id AS fact_id, "
            "CASE WHEN json_valid(m.metadata_json) "
            "THEN json_extract(m.metadata_json, '$.project') END AS project "
            "FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id "
            f"WHERE f.fact_id IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        for row in rows:
            d = dict(row)
            if isinstance(d.get("project"), str) and d["project"].strip():
                found[str(d["fact_id"])] = d["project"]
    return found


def project_keys(db: Any, fact_ids: Iterable[str]) -> dict[str, str]:
    """``{fact_id: project_key}`` for the facts saved under a project."""
    out: dict[str, str] = {}
    for fid, raw in stored_projects(db, fact_ids).items():
        key = project_key(raw)
        if key is not None:
            out[fid] = key
    return out


@dataclass(frozen=True, slots=True)
class Scoped:
    """What narrowing left: the kept ids (incoming order), the ones saved under
    the preferred project, and the report for the response (None when the
    caller named no project)."""

    kept: tuple[str, ...]
    preferred: frozenset[str] = frozenset()
    report: dict | None = None


def _filter_note(raw: str, key: str | None, reason: str) -> str:
    if reason == "not_a_project":
        return f"'{raw}' does not name a project, so these results are not narrowed."
    if reason == "unreadable":
        return (f"The project saved with each memory could not be read, so these "
                f"results are not narrowed to project '{key}'.")
    return (f"None of the memories found for this question were saved under project "
            f"'{key}', so these results are not narrowed to it.")


def _apply_filter(ids: list[str], raw: str, keys: dict[str, str] | None
                  ) -> tuple[list[str], dict]:
    want = project_key(raw)
    if want is None:
        reason, matched = "not_a_project", []
    elif keys is None:
        reason, matched = "unreadable", []
    else:
        matched = [f for f in ids if keys.get(f) == want]
        reason = "" if matched else "no_match"
    report = {"project": raw, "key": want, "applied": bool(matched),
              "matched": len(matched), "note": ""}
    if not matched:
        report = {**report, "reason": reason, "note": _filter_note(raw, want, reason)}
        return ids, report
    return matched, report


def _prefer(ids: list[str], raw: str, keys: dict[str, str] | None
            ) -> tuple[frozenset[str], dict]:
    want = project_key(raw)
    preferred = (frozenset(f for f in ids if keys.get(f) == want)
                 if want is not None and keys is not None else frozenset())
    report = {"project": raw, "key": want, "boost": BOOST, "matched": len(preferred)}
    if want is None:
        report["note"] = f"'{raw}' does not name a project, so no memory was preferred."
    elif keys is None:
        report["note"] = "The project saved with each memory could not be read."
    return preferred, report


def narrow(db: Any, fact_ids: Iterable[str], profile_id: str, facets: Any, *,
           resolver: Any = None, display_min_confidence: float | None = None) -> Scoped:
    """Apply the caller's facets to ``fact_ids`` (already admitted candidates).

    ``project`` filters with the fall-back described above; ``saved_by``,
    ``about`` and ``kind`` stay hard filters exactly as before (they keep
    nothing when they match nothing, or when they cannot be checked);
    ``prefer_project`` only marks which kept ids get the boost.
    """
    from superlocalmemory.retrieval.facets import matching_fact_ids

    ids = list(dict.fromkeys(str(f) for f in fact_ids))
    wants_project = getattr(facets, "project", None) is not None
    wants_prefer = getattr(facets, "prefer_project", None) is not None
    keys: dict[str, str] | None = None
    if (wants_project or wants_prefer) and ids:
        try:
            keys = project_keys(db, ids)
        except Exception as exc:  # noqa: BLE001 - reported, never silent
            logger.warning("Recall project lookup failed (%s)", type(exc).__name__)
            keys = None
    elif wants_project or wants_prefer:
        keys = {}
    report: dict[str, dict] = {}
    if wants_project:
        ids, report["filter"] = _apply_filter(ids, facets.project, keys)
    rest = replace(facets, project=None, prefer_project=None)
    if rest.narrows and ids:
        kwargs = ({} if display_min_confidence is None
                  else {"display_min_confidence": display_min_confidence})
        keep = matching_fact_ids(db, ids, profile_id, rest, resolver=resolver, **kwargs)
        ids = [f for f in ids if f in keep]
    preferred: frozenset[str] = frozenset()
    if wants_prefer:
        preferred, report["prefer"] = _prefer(ids, facets.prefer_project, keys)
    return Scoped(kept=tuple(ids), preferred=preferred, report=report or None)


def boost_order(items: Sequence[Any], preferred: frozenset[str], *,
                score: Any = None, ident: Any = None) -> list:
    """``items`` re-ordered with same-project ones lifted by BOOST.

    ``score(item)`` and ``ident(item)`` default to a FusionResult's
    ``fused_score`` and ``fact_id``. Same items, nothing added or removed; the
    incoming order breaks ties, so the result is repeatable.
    """
    if not preferred or not items:
        return list(items)
    score = score or (lambda it: it.fused_score)
    ident = ident or (lambda it: it.fact_id)
    keyed = sorted(
        range(len(items)),
        key=lambda i: (-(lift(score(items[i])) if ident(items[i]) in preferred
                         else float(score(items[i]))), i),
    )
    return [items[i] for i in keyed]


__all__ = ["BOOST", "Scoped", "boost_order", "lift", "narrow", "project_keys",
           "stored_projects"]
