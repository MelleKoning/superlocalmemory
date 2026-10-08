# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What a recall's ``tags`` filter did, in words (4.1.22).

``tags`` is a hard filter (``retrieval/facets``): unlike ``project``
(``retrieval/project_scope``) it never falls back to unfiltered results when
nothing matches — a tag filter that finds nothing really did find nothing,
and that answer is correct. But "nothing matched" is ambiguous on its own: it
could mean nobody ever saved a memory with that tag at all, or it could mean
some memory somewhere does carry it but none of the memories this question's
other filters and candidates admitted did. Those are different facts about
the store, and conflating them is the same defect ``project_scope`` exists to
avoid for projects — "an unmatched exact filter must be distinguishable from
nothing was ever stored" (4.1.22).

This module answers that one question, only when it is needed (``matched``
is already free — ``retrieval.facets.matching_fact_ids`` computed it as part
of the ordinary filter pass). When the tag filter left something, no further
read happens. When it left nothing, one profile-wide scan
(``retrieval.tag_search.tag_members``, personal scope — the same reach the
filter and the search-inside supplement both have) answers whether ANY
visible memory carries the requested tag(s), under the SAME all/any
semantics the filter itself used.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _describe(tags: tuple[str, ...], match: str) -> str:
    quoted = [f"'{t}'" for t in tags]
    if len(quoted) == 1:
        return f"the tag {quoted[0]}"
    joiner = " or " if match == "any" else " and "
    if len(quoted) == 2:
        return "the tags " + joiner.join(quoted)
    return "the tags " + ", ".join(quoted[:-1]) + f"{joiner}{quoted[-1]}"


def build_report(db: Any, profile_id: str, facets: Any, matched: int) -> dict:
    """``tag_scope`` for a recall response: what the ``tags`` filter did.

    ``matched`` is the number of candidates that survived every facet
    (project/saved_by/about/kind/tags combined) — the caller already has it
    from ``retrieval.project_scope.narrow``'s result; this never re-derives
    it, so the common (non-empty) case costs nothing extra.

    Always has ``applied: True`` — unlike ``project_scope``, a tag filter
    never un-applies itself; an empty ``matched`` IS its answer, not a
    fallback trigger. ``reason``/``note`` are only present when ``matched``
    is 0, and distinguish:

    * ``"no_memory_has_tag"`` — no visible memory in this profile carries the
      requested tag(s) at all, under the same all/any semantics asked for;
    * ``"none_relevant"`` — some memory does, but it did not survive this
      question's other filters or was not among its candidates;
    * ``"unreadable"`` — the profile-wide check itself could not run (DB
      error); reported rather than silently guessed at.
    """
    from superlocalmemory.core.tag_identity import tag_key

    tags = tuple(getattr(facets, "tags", ()) or ())
    match = getattr(facets, "tags_match", "all") or "all"
    keys = sorted({k for k in (tag_key(t) for t in tags) if k is not None})
    report: dict[str, Any] = {
        "tags": list(tags), "keys": keys, "match": match,
        "applied": True, "matched": int(matched), "note": "",
    }
    if matched > 0 or not keys:
        return report
    from superlocalmemory.retrieval.tag_search import tag_members

    try:
        # tag_members never raises: None is its "could not read", which must
        # be reported as unreadable, not as "no memory has the tag".
        found = tag_members(db, profile_id, tags, match)
        exists = None if found is None else bool(found)
    except Exception as exc:  # noqa: BLE001 — reported, never silent
        logger.warning("tag_scope existence check failed (%s)", type(exc).__name__)
        exists = None
    label = _describe(tags, match)
    if exists is None:
        report["reason"] = "unreadable"
        report["note"] = f"Whether any memory carries {label} could not be checked."
    elif exists:
        report["reason"] = "none_relevant"
        report["note"] = (f"Some memories carry {label}, but none matched this "
                          "question's other filters or candidates.")
    else:
        report["reason"] = "no_memory_has_tag"
        report["note"] = f"No memory in this profile is saved with {label}."
    return report


__all__ = ["build_report"]
