# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A correction the ledger refuses is a conflict, never "temporarily unavailable".

The sole writer turns any exception a command raises into a generic writer
failure, which the mutation runtime then reports as an outage (HTTP 503) that
clients retry. A correction the ledger refused because its case changed, or
because a memory it names is being deleted, will never succeed on retry, so it
is reported as the conflict it is (HTTP 409), with the ledger's own reason.
"""

from __future__ import annotations


def _ledger_refusal(exc: BaseException | None) -> BaseException | None:
    from superlocalmemory.storage.correction_cases import CorrectionCompareAndSetError

    seen = 0
    while exc is not None and seen < 8:  # bounded walk down the cause chain
        if isinstance(exc, CorrectionCompareAndSetError):
            return exc
        exc, seen = exc.__cause__, seen + 1
    return None


def refusal_or_outage(exc: BaseException) -> Exception:
    """What a failed mutation submission raises to its caller."""
    from superlocalmemory.core.remember_runtime import (
        CanonicalMutationConflict,
        CanonicalRememberUnavailable,
    )

    refused = _ledger_refusal(exc)
    if refused is not None:
        return CanonicalMutationConflict(str(refused))
    return CanonicalRememberUnavailable("canonical mutation is temporarily unavailable")


__all__ = ["refusal_or_outage"]
