# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""An edit of a memory that already has a correction open is refused truthfully.

Editing a memory proposes a correction that a person reviews. The ledger keeps
at most one open correction per memory (``uq_correction_cases_active_
predecessor``: one ``proposed`` or ``applied`` case). 4.1.21 let a second edit
run into that rule inside the writer, where the failed insert was reported as
"canonical mutation writer is temporarily unavailable" (HTTP 503): a refusal
that will never change, presented as an outage worth retrying, so every
client retried it. Measured on a 22k-fact store copy: every second edit of
the same memory, 40 of 70 edits in one run.

The edit is now refused up front, inside the same writer transaction (so the
check and the insert cannot race), as a conflict that names the open case and
says what to do instead.
"""

from __future__ import annotations

from typing import Any


def refuse_second_open_case(connection: Any, profile_id: str, fact_id: str) -> None:
    """Raise a conflict when ``fact_id`` already has a proposed or applied case.

    A case only SLM itself proposed and nobody reviewed does not count: the
    user's edit overtakes it in this same transaction (core/overtaken_cases,
    Varun's "user action wins"). One rule decides both, so they cannot drift.
    """
    from superlocalmemory.core.overtaken_cases import is_machine_pending

    cursor = connection.execute(
        "SELECT case_id, status, successor_fact_id, reason_code, proposed_by_actor_kind "
        "FROM correction_cases WHERE profile_id = ? AND predecessor_fact_id = ? "
        "AND status IN ('proposed', 'applied') ORDER BY created_at, case_id",
        (profile_id, fact_id),
    )
    rows = [r for r in cursor.fetchall()
            if not is_machine_pending({"status": r[1], "reason_code": r[3],
                                       "proposed_by_actor_kind": r[4]})]
    if not rows:
        return
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    case_id, status, successor = rows[0][0], rows[0][1], rows[0][2]
    if status == "proposed":
        raise CanonicalMutationConflict(
            f"memory {fact_id} already has a correction waiting for review "
            f"(case {case_id}); apply or reject that case before editing it again"
        )
    raise CanonicalMutationConflict(
        f"memory {fact_id} was already corrected (case {case_id}); "
        f"edit its current version {successor} instead"
    )


__all__ = ["refuse_second_open_case"]
