# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Withhold or flag a derived fact that changes what its source memory said.

Runs where a remember is checkpointed, for every final fact that is not an
exact span of the raw text the caller sent.

* A fact that turns a source number into a date (``number_became_date``, the
  proven 4.1.21 harm: "2004.6 ms" stored as June 2004) is withheld
  (``quarantined = 1``); its lineage reason is ``source_fidelity:<reasons>``.
* Every other finding — a date or measurement the source never stated, a
  dropped "never", a reversed order, a lost "previously" — is a heuristic with
  known faithful-paraphrase hits ("is not enabled" -> "is disabled"). The fact
  stays in answers; its lineage reason is ``source_fidelity_unverified:<reasons>``
  and recall marks it unverified (``retrieval/source_fidelity_flags.py``).

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
#: Prefix for a fact that stays in answers, marked unverified.
FLAGGED_PREFIX = "source_fidelity_unverified:"
#: The only findings precise enough to take a fact out of answers.
WITHHOLD_REASONS = frozenset({"number_became_date"})


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


def judge_fidelity(
    db: Any,
    *,
    profile_id: str,
    fact_id: str,
    content: str,
    raw_content: str,
) -> tuple[str | None, bool]:
    """``(lineage reason or None, withhold?)``, with reads only - nothing written.

    Split from the write so the check can run before the checkpoint's write
    transaction opens (``core/derivation_lineage.plan_operation_lineage``);
    ``withhold`` is applied inside it by ``withhold_fact``. Never raises.
    """
    try:
        report = check_fact_against_source(content, raw_content)
    except Exception as exc:  # pragma: no cover - defensive, pure function
        logger.warning("source fidelity check failed for %s: %s", fact_id[:16], exc)
        return None, False
    if report.ok:
        return None, False
    released = _released_reason(db, profile_id, fact_id)
    if released:
        return released, False  # the user judged it correct; never withhold it again
    if not WITHHOLD_REASONS & set(report.reasons):
        return FLAGGED_PREFIX + "+".join(report.reasons), False
    return REASON_PREFIX + "+".join(report.reasons), True


def is_released(db: Any, *, profile_id: str, fact_id: str) -> bool:
    """Whether the user has released this fact (read inside the write, to be sure)."""
    return _released_reason(db, profile_id, fact_id) is not None


def withhold_fact(db: Any, *, profile_id: str, fact_id: str, reason: str) -> None:
    """Take a fact out of answers (``quarantined = 1``); reversible, loses nothing."""
    if _has_quarantine_column(db):
        db.execute(
            "UPDATE atomic_facts SET quarantined = 1 "
            "WHERE fact_id = ? AND profile_id = ? AND COALESCE(quarantined, 0) = 0",
            (fact_id, profile_id),
        )
    logger.info(
        "Withheld derived fact %s: it does not match its source (%s)",
        fact_id[:16], reason[len(REASON_PREFIX):].replace("+", ","),
    )


def withhold_if_unfaithful(
    db: Any,
    *,
    profile_id: str,
    fact_id: str,
    content: str,
    raw_content: str,
) -> str | None:
    """Return the lineage reason when the fact was withheld or flagged, else None.

    Never raises: a failure to check leaves the fact as it was and is logged,
    because a fidelity check must never cost the user a write.
    """
    reason, withhold = judge_fidelity(db, profile_id=profile_id, fact_id=fact_id,
                                      content=content, raw_content=raw_content)
    if withhold and reason:
        withhold_fact(db, profile_id=profile_id, fact_id=fact_id, reason=reason)
    return reason


__all__ = ["FLAGGED_PREFIX", "REASON_PREFIX", "RELEASED_PREFIX", "WITHHOLD_REASONS",
           "is_released", "judge_fidelity", "withhold_fact", "withhold_if_unfaithful"]
