# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Which correction cases keep a memory from being forgotten, asked up front.

Correction history is immutable on purpose (M042): both facts of a case are
referenced ``ON DELETE RESTRICT`` and no case is ever removed, whatever its
status. So a fact named by any case cannot be forgotten with an ordinary delete.

4.1.21 found that out too late. The delete route erased the fact's search
indexes and wrote its deletion record first, and only then did the canonical
delete refuse. The memory stayed stored, was reported "protected", and could no
longer be found by its words or its meaning. Asking this question before any
destructive step keeps a refusal a refusal.
"""

from __future__ import annotations

from typing import Any

_MAX_NAMED = 3


def _named(rows: list[dict], profile_id: str) -> list[tuple[str, str]]:
    return [(str(r["case_id"]) if str(r["profile_id"]) == profile_id else "(another workspace)",
             str(r["status"])) for r in rows]


def protecting_cases(db: Any, profile_id: str, fact_id: str) -> list[tuple[str, str]]:
    """``(case_id, status)`` of every case naming the fact, oldest first.

    Matches the foreign key exactly: it restricts by fact id whatever profile
    the case is filed under. A case from another profile is not named here
    (``"(another workspace)"``).
    """
    from superlocalmemory.core.overtaken_cases import cases_naming

    return _named(cases_naming(db, [fact_id]), profile_id)


def blocking_cases(db: Any, profile_id: str, fact_id: str) -> list[tuple[str, str]]:
    """The cases that still refuse a user's delete: proposed by a person, or
    already decided. A pending machine proposal is overtaken by the user's
    action instead (core/overtaken_cases.py)."""
    from superlocalmemory.core.overtaken_cases import blocking, cases_naming

    return _named(blocking(cases_naming(db, [fact_id])), profile_id)


def protection_message(cases: list[tuple[str, str]]) -> str:
    """The true reason, naming the cases, without advice that cannot work.

    Rejecting or rolling back a case does not release the fact: the history
    stays, so the message must not tell anyone to resolve it first.
    """
    named = ", ".join(f"{case_id} ({status})" for case_id, status in cases[:_MAX_NAMED])
    more = f" and {len(cases) - _MAX_NAMED} more" if len(cases) > _MAX_NAMED else ""
    return (
        "fact is protected by correction history: it is part of correction case "
        f"{named}{more}, which a person proposed or which was already decided, and "
        "correction history is kept on purpose, so this memory cannot be deleted. "
        "Nothing was changed."
    )


__all__ = ["blocking_cases", "protecting_cases", "protection_message"]
