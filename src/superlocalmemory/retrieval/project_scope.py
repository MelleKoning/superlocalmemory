# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Using the project a recall is about (GitHub #150).

Two ways to say it, both opt-in per recall:

* ``prefer_project`` - a soft preference. Memories saved under that project
  rank above others of *similar* relevance; nothing is removed, and memories
  saved with no project (most of any store written before 4.1.21) stay
  exactly as findable as before. ``session_init`` passes the session's project
  this way. It acts twice, each time bounded by BOOST: on which candidates
  make the cut (``boost_order``, inside retrieval) and on the final order
  (``prefer_in_final_order``, the last ranking step of recall, after learned
  ranking - which rewrites every score without knowing the project).
* ``project`` - a filter. Only memories saved under that project are kept,
  and recall also searches inside the project before results are fused
  (``retrieval.project_search``), so a project memory crowded out by closer
  matches elsewhere is still a candidate.
  When none of the memories found for the question were saved under it - a
  project with no tagged memories, a misspelt name - the filter is not applied,
  the unfiltered results are returned, and the response says so
  (``project_scope.filter.applied: false`` with a plain-English ``note``). A
  filter that silently returns nothing is indistinguishable from "this store
  knows nothing", which is the wrong answer. ``project_strict`` (4.1.22) turns
  the fall-back off: nothing is returned, and the response says why. For a
  caller (an automation, a loop) that must never act on another project's
  memories. A project label is never an access grant: profile and scope
  admission, which ran before this, are the boundary.

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

from superlocalmemory.core.project_identity import project_key, project_name

logger = logging.getLogger(__name__)

#: How much a same-project memory's ranking score is lifted: x(1 + BOOST).
#:
#: Why 0.25. The bound is the point: a same-project memory passes another
#: memory only when its own score is already at least 1/1.25 = 80% of that
#: memory's. A weakly relevant project memory can never jump a strongly
#: relevant one - at two thirds of the other's score it stays below it.
#: Measured on the ranking fixture (tests/test_retrieval/
#: test_project_scope_ranking.py): the on-topic project memories that
#: other memories outranked sat at 0.84-0.98 of that memory's ranking
#: score, so 0.25 puts every one of them first; one project memory at 0.77
#: of a better match stays below it, which is the bound doing its job. Half the
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


def _filter_note(raw: str, key: str | None, reason: str, strict: bool = False) -> str:
    if strict:
        if reason == "not_a_project":
            return f"'{raw}' does not name a project, so nothing is returned (strict)."
        if reason == "unreadable":
            return (f"The project saved with each memory could not be read, so nothing "
                    f"is returned (strict project '{key}').")
        return (f"Nothing saved under project '{key}' was found for this question, "
                f"so nothing is returned (strict).")
    if reason == "not_a_project":
        return f"'{raw}' does not name a project, so these results are not narrowed."
    if reason == "unreadable":
        return (f"The project saved with each memory could not be read, so these "
                f"results are not narrowed to project '{key}'.")
    return (f"None of the memories found for this question were saved under project "
            f"'{key}', so these results are not narrowed to it.")


#: How two project values are judged the same (core.project_identity).
MATCH_RULE = "last folder name, ignoring case"


def _identity(raw: str, want: str | None, matched: list[str],
              stored: dict[str, str] | None) -> dict:
    """What the filter matched against, and whether one name stood for more
    than one stored project ("/a/app" and "/b/app" are the same key)."""
    out: dict = {"name": project_name(raw), "rule": MATCH_RULE}
    if stored and matched:
        variants = sorted({stored[f].strip() for f in matched if f in stored})
        if len({v.rstrip("/\\").casefold() for v in variants}) > 1:
            out["ambiguous"] = True
            out["stored_as"] = variants[:3]
            out["stored_as_count"] = len(variants)
            out["ambiguity_note"] = (
                f"Memories saved as {len(variants)} different project values share the "
                f"name '{want}'; they are treated as one project.")
    return out


def _apply_filter(ids: list[str], raw: str, keys: dict[str, str] | None,
                  *, strict: bool = False, stored: dict[str, str] | None = None
                  ) -> tuple[list[str], dict]:
    want = project_key(raw)
    if want is None:
        reason, matched = "not_a_project", []
    elif keys is None:
        reason, matched = "unreadable", []
    else:
        matched = [f for f in ids if keys.get(f) == want]
        reason = "" if matched else "no_match"
    report = {"project": raw, "key": want, "applied": bool(matched) or strict,
              "strict": strict, "matched": len(matched), "note": "",
              "identity": _identity(raw, want, matched, stored)}
    if not matched:
        report = {**report, "reason": reason,
                  "note": _filter_note(raw, want, reason, strict)}
        return ([] if strict else ids), report
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
    stored: dict[str, str] | None = None
    if (wants_project or wants_prefer) and ids:
        try:
            stored = stored_projects(db, ids)
            keys = {f: k for f, k in ((f, project_key(v)) for f, v in stored.items())
                    if k is not None}
        except Exception as exc:  # noqa: BLE001 - reported, never silent
            logger.warning("Recall project lookup failed (%s)", type(exc).__name__)
            keys = stored = None
    elif wants_project or wants_prefer:
        keys = stored = {}
    report: dict[str, dict] = {}
    if wants_project:
        ids, report["filter"] = _apply_filter(
            ids, facets.project, keys,
            strict=bool(getattr(facets, "project_strict", False)), stored=stored)
    rest = replace(facets, project=None, prefer_project=None, project_strict=False)
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


def _utility(result: Any) -> float:
    from superlocalmemory.core.recall_pipeline import _rank_key

    return -_rank_key(result)[0]


def prefer_in_final_order(results: list, preferred: frozenset[str]) -> list:
    """The finished answer with same-project results lifted, as recall's
    last ordering step - after learned ranking, which rewrites every ranking
    score and knows nothing of projects, so a preference applied only inside
    retrieval was undone on every store with learning switched on.

    The same bound, on the final ranking key (``recall_pipeline._rank_key``):
    a same-project result moves up past a neighbour only while that
    neighbour's key is below its own lifted by BOOST, and never past another
    same-project result. Every other result keeps its relative order. Moved
    or not, a same-project result carries "same_project" in its evidence and
    its lifted key, so a later bounded pass (kind awareness) compares against
    what decided this order. New objects; the input is not modified.
    """
    if not preferred or not results:
        return results
    ids = [getattr(getattr(r, "fact", None), "fact_id", None) for r in results]
    marked = [fid in preferred for fid in ids]
    if not any(marked):
        return results
    keys = [_utility(r) for r in results]
    lifted = [lift(k) if m else k for k, m in zip(keys, marked)]
    order = list(range(len(results)))
    for item in range(len(results)):
        if not marked[item]:
            continue
        pos = order.index(item)
        while pos > 0:
            above = order[pos - 1]
            if marked[above] or keys[above] >= lifted[item]:
                break
            order[pos - 1], order[pos] = item, above
            pos -= 1
    out = []
    for rank, i in enumerate(order, start=1):
        r = results[i]
        if marked[i]:
            chain = list(r.evidence_chain or [])
            r = replace(r, ranking_score=lifted[i], rank_position=rank,
                        evidence_chain=chain + ([] if "same_project" in chain
                                                else ["same_project"]))
        else:
            r = replace(r, rank_position=rank)
        out.append(r)
    return out


def preferred_in(db: Any, results: list, facets: Any) -> frozenset[str]:
    """Which of ``results`` were saved under ``facets.prefer_project``.
    Empty when no preference was asked for, it names no project, or the store
    cannot be read (logged; the order is then simply left alone)."""
    want = project_key(getattr(facets, "prefer_project", None))
    if want is None or not results:
        return frozenset()
    ids = [r.fact.fact_id for r in results if getattr(r, "fact", None) is not None]
    try:
        keys = project_keys(db, ids)
    except Exception as exc:  # noqa: BLE001 - ordering is advisory, reported
        logger.warning("Recall project lookup for the final order failed (%s)",
                       type(exc).__name__)
        return frozenset()
    return frozenset(fid for fid, key in keys.items() if key == want)


__all__ = ["BOOST", "Scoped", "boost_order", "lift", "narrow", "preferred_in",
           "prefer_in_final_order", "project_keys", "stored_projects"]
