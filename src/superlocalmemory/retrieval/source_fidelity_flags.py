# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Mark a recall result whose fact may not say what its memory said.

A derived fact that fails the source check (``encoding/source_fidelity.py``)
but is not withheld — an unstated date or number, a dropped "never", a
reversed order, a lost "previously" — is still returned, with
``source_fidelity_unverified:<reasons>`` appended to its ``evidence_chain``.
That is the per-result provenance list every recall surface already carries
(HTTP ``/recall``, MCP ``recall``, the CLI), so a caller sees the flag without
a new response field. Old facts written before the check existed are marked
the same way, because the check runs here against the parent memory.

Read-only, one indexed query, no model. Order and scores are untouched. A fact
the user released with ``slm db fidelity --release`` is not marked. Any error
returns the results unchanged: a flag must never cost the user a recall.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from superlocalmemory.core.source_fidelity_guard import FLAGGED_PREFIX, RELEASED_PREFIX
from superlocalmemory.encoding.source_fidelity import check_fact_against_source

logger = logging.getLogger(__name__)


def _unverified(db: Any, profile_id: str, fact_ids: list[str]) -> dict[str, str]:
    marks = ",".join("?" for _ in fact_ids)
    rows = db.execute(
        "SELECT f.fact_id, f.content, m.content AS source FROM atomic_facts f "
        "JOIN memories m ON m.memory_id = f.memory_id "
        f"WHERE f.profile_id = ? AND f.fact_id IN ({marks})",
        (profile_id, *fact_ids),
    )
    released = {
        str(dict(r)["object_id"]) for r in db.execute(
            "SELECT object_id FROM derivation_lineage WHERE profile_id = ? "
            f"AND object_type = 'fact' AND object_id IN ({marks}) "
            "AND unresolved_reason LIKE ?",
            (profile_id, *fact_ids, RELEASED_PREFIX + "%"),
        )
    }
    flags: dict[str, str] = {}
    for row in rows:
        data = dict(row)
        fact_id, fact, source = str(data["fact_id"]), data["content"] or "", data["source"] or ""
        if fact_id in released or not fact or not source or fact.strip() in source:
            continue
        report = check_fact_against_source(fact, source)
        if not report.ok:
            flags[fact_id] = FLAGGED_PREFIX + "+".join(report.reasons)
    return flags


def flag_unverified(results: list, db: Any, profile_id: str) -> list:
    """``results`` with an unverified-source label on each fact that fails the check."""
    try:
        fact_ids = [r.fact.fact_id for r in results if getattr(r, "fact", None) is not None]
        if not fact_ids or db is None:
            return results
        flags = _unverified(db, profile_id, fact_ids)
    except Exception as exc:  # noqa: BLE001 - fail open, never cost a recall
        logger.debug("source fidelity flags skipped: %s", type(exc).__name__)
        return results
    if not flags:
        return results
    out = []
    for r in results:
        label = flags.get(getattr(getattr(r, "fact", None), "fact_id", ""), "")
        chain = list(getattr(r, "evidence_chain", None) or [])
        out.append(replace(r, evidence_chain=[*chain, label]) if label and label not in chain
                   else r)
    return out


__all__ = ["flag_unverified"]
