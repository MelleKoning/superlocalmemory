# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Withhold a derived fact that changes what its source memory said.

Runs where a remember is checkpointed, for every final fact that is not an
exact span of the raw text the caller sent. A fact that turns a number into a
date, introduces a date or measurement the source never stated, or drops a
"never" / reverses an order / loses a "previously" is not stored as a trusted
answer: it is withheld (``quarantined = 1``) and the reason is written to its
derivation lineage as ``source_fidelity:<reasons>``.

Withholding is reversible and loses nothing. The row, its text and its
provenance stay on disk; the memory's own verbatim fact, which is an exact
copy of what the caller sent, stays searchable. ``slm db fidelity`` lists what
was withheld and releases a fact the user judges correct.
"""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.encoding.source_fidelity import check_fact_against_source

logger = logging.getLogger(__name__)

#: Prefix of ``derivation_lineage.unresolved_reason`` for a withheld fact.
REASON_PREFIX = "source_fidelity:"
#: Prefix once the user released it (``slm db fidelity --release``).
RELEASED_PREFIX = "source_fidelity_released:"


def _released_reason(db: Any, profile_id: str, fact_id: str) -> str | None:
    try:
        rows = db.execute(
            "SELECT unresolved_reason FROM derivation_lineage "
            "WHERE profile_id = ? AND object_type = 'fact' AND object_id = ? "
            "AND unresolved_reason LIKE ? LIMIT 1",
            (profile_id, fact_id, RELEASED_PREFIX + "%"),
        )
    except Exception:
        return None
    return str(dict(rows[0])["unresolved_reason"]) if rows else None


def _has_quarantine_column(db: Any) -> bool:
    try:
        rows = db.execute("PRAGMA table_info(atomic_facts)")
    except Exception:
        return False
    return any(dict(row).get("name") == "quarantined" for row in rows)


def withhold_if_unfaithful(
    db: Any,
    *,
    profile_id: str,
    fact_id: str,
    content: str,
    raw_content: str,
) -> str | None:
    """Return the lineage reason when the fact was withheld, else None.

    Never raises: a failure to check leaves the fact as it was and is logged,
    because a fidelity check must never cost the user a write.
    """
    try:
        report = check_fact_against_source(content, raw_content)
    except Exception as exc:  # pragma: no cover - defensive, pure function
        logger.warning("source fidelity check failed for %s: %s", fact_id[:16], exc)
        return None
    if report.ok:
        return None
    released = _released_reason(db, profile_id, fact_id)
    if released:
        return released  # the user judged it correct; never withhold it again
    reason = REASON_PREFIX + "+".join(report.reasons)
    if _has_quarantine_column(db):
        db.execute(
            "UPDATE atomic_facts SET quarantined = 1 "
            "WHERE fact_id = ? AND profile_id = ? AND COALESCE(quarantined, 0) = 0",
            (fact_id, profile_id),
        )
    logger.info(
        "Withheld derived fact %s: it does not match its source (%s)",
        fact_id[:16], ",".join(report.reasons),
    )
    return reason


__all__ = ["REASON_PREFIX", "RELEASED_PREFIX", "withhold_if_unfaithful"]
