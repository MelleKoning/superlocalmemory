# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Journal-first service boundary for a durable, immediately-queryable remember."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    AdmissionJournalUnavailable,
    PreparedAdmission,
    RememberRequest,
    TerminalAdmissionError,
)
from superlocalmemory.storage.write_coordinator import (
    QueueOverloadedError,
    WriteDeadlineExceededError,
    WriterStalledError,
)


class AdmissionRejected(RuntimeError):
    """Canonical admission was not committed; the journal records its outcome."""

    def __init__(self, error_code: str, *, retryable: bool = False) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RememberReceipt:
    """Canonical receipt returned only after admission is queryable."""

    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, receipt: Mapping[str, Any]) -> RememberReceipt:
        return cls(payload=dict(receipt))


@dataclass(frozen=True, slots=True)
class RememberAdmissionCommand:
    """Typed foreground input for a coordinator's bounded canonical transaction."""

    journal_id: str
    request_hash: str
    request: RememberRequest
    profile_id: str
    idempotency_key: str

    @classmethod
    def from_prepared(
        cls, prepared: PreparedAdmission, request: RememberRequest
    ) -> RememberAdmissionCommand:
        return cls(
            journal_id=prepared.journal_id,
            request_hash=prepared.request_hash,
            request=request,
            profile_id=prepared.profile_id,
            idempotency_key=prepared.idempotency_key,
        )


class RememberCoordinator(Protocol):
    def submit(self, command: RememberAdmissionCommand, *, wait_ms: int) -> Any: ...


def accepted_receipt(entry: PreparedAdmission) -> RememberReceipt:
    """The receipt for a memory that is durable but not yet searchable.

    It says exactly that and nothing more: no fact ids exist yet, so none are
    claimed. Resending the same idempotency key returns the canonical receipt
    once the commit lands, and never stores the memory twice.
    """
    return RememberReceipt(
        payload={
            "status": "accepted",
            "materialization_state": "accepted",
            "durable": True,
            "queryable": False,
            "admission_id": entry.journal_id,
            "idempotency_key": entry.idempotency_key,
            "operation_id": None,
            "pending_id": None,
            "commit_sequence": None,
            "fact_ids": [],
            "count": 0,
        }
    )


#: Failures that mean "the canonical writer is busy right now", as opposed to
#: "the canonical writer is broken". Only these may turn into an accepted
#: receipt; anything else stays a refusal.
_CONTENTION = (
    AdmissionJournalUnavailable,
    QueueOverloadedError,
    WriteDeadlineExceededError,
    WriterStalledError,
)


class RememberService:
    """Apply the journaling pattern without doing enrichment inside admission."""

    def __init__(self, journal: AdmissionJournal, coordinator: RememberCoordinator) -> None:
        self._journal = journal
        self._coordinator = coordinator

    def remember(
        self,
        request: RememberRequest,
        actor: Actor,
        *,
        deadline_ms: int,
        defer: Callable[[PreparedAdmission], None] | None = None,
    ) -> RememberReceipt:
        """Journal, then commit within the deadline.

        ``defer`` is the contention path. Once the journal holds the request
        it is durable: if the canonical writer is merely busy past the
        deadline, the entry is handed to ``defer`` (which finishes the commit)
        and an ``accepted`` receipt is returned instead of an error. Without
        ``defer`` the old behaviour holds and contention raises.
        """
        if deadline_ms <= 0:
            raise ValueError("deadline_ms must be greater than zero")
        deadline = time.monotonic() + deadline_ms / 1_000
        prepared = self._journal.prepare(
            request,
            actor,
            deadline=deadline,
        )
        if prepared.original_receipt is not None:
            return RememberReceipt.from_mapping(prepared.original_receipt)
        if prepared.state == "rejected":
            raise AdmissionRejected(prepared.error_code or "COMMAND_REJECTED")
        try:
            return self._commit_prepared(prepared, deadline)
        except _CONTENTION:
            if defer is None:
                raise
            # The cancelled coordinator item never reached its commit (the
            # coordinator waits through a commit once it has started), so the
            # journal entry is the only copy and the deferred commit is the
            # only writer of it. Nothing is lost and nothing is doubled.
            defer(prepared)
            return accepted_receipt(prepared)

    def _commit_prepared(
        self, prepared: PreparedAdmission, deadline: float,
    ) -> RememberReceipt:
        command_request = self._journal.request_for(
            prepared,
            deadline=deadline,
        )
        dispatched = self._journal.mark_dispatched(
            prepared.journal_id,
            deadline=deadline,
            known_prepared=prepared.state == "prepared",
        )
        if dispatched.original_receipt is not None:
            return RememberReceipt.from_mapping(dispatched.original_receipt)
        try:
            result = self._coordinator.submit(
                RememberAdmissionCommand.from_prepared(prepared, command_request),
                wait_ms=_remaining_milliseconds(deadline),
            )
        except TerminalAdmissionError as exc:
            try:
                self._journal.mark_rejected(
                    prepared.journal_id,
                    exc.error_code,
                    deadline=deadline,
                )
            except AdmissionJournalUnavailable:
                # A rejection is final whether or not the journal records it
                # in time; replay re-derives it. It must never be reported as
                # contention, which would answer "accepted" for a refusal.
                pass
            raise AdmissionRejected(exc.error_code) from exc
        state = _result_value(result, "state")
        receipt = _result_value(result, "receipt") or {}
        if state in {"committed", "duplicate"}:
            if not isinstance(receipt, Mapping):
                raise AdmissionRejected("COMMAND_REJECTED: canonical result had no receipt")
            try:
                committed = self._journal.mark_committed(
                    prepared.journal_id,
                    receipt,
                    deadline=deadline,
                )
            except AdmissionJournalUnavailable:
                # The canonical receipt is already durable and idempotent. Do
                # not turn that committed write into an ambiguous client
                # failure merely because the auxiliary journal exhausted the
                # caller's remaining budget. The dispatched record is safe for
                # retry/replay, which will recover the same immutable receipt.
                return RememberReceipt.from_mapping(receipt)
            return RememberReceipt.from_mapping(committed.original_receipt or receipt)

        error_code = str(_result_value(result, "error_code") or "COMMAND_REJECTED")
        if state == "rejected":
            try:
                self._journal.mark_rejected(
                    prepared.journal_id,
                    error_code,
                    deadline=deadline,
                )
            except AdmissionJournalUnavailable:
                pass  # final regardless; see the terminal branch above
        raise AdmissionRejected(error_code, retryable=state != "rejected")


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(key)
    return getattr(result, key, None)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AdmissionJournalUnavailable("remember admission deadline expired")
    return remaining


def _remaining_milliseconds(deadline: float) -> int:
    return max(1, math.ceil(_remaining_seconds(deadline) * 1_000))
