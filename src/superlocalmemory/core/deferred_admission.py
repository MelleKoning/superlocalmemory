# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Finish remembers that were accepted while the canonical writer was busy.

A remember is durable the moment the admission journal holds it. When the
canonical writer cannot commit it inside the caller's deadline — another
connection holds ``memory.db``'s write lock, the coordinator queue is deep —
the caller is told the truth ("accepted, not yet searchable") and the journal
entry is handed here. One thread commits deferred entries in arrival order,
retrying contention with backoff until each one is committed or deterministically
rejected.

Exactly-once rests on two existing guarantees, not on this module: the
coordinator dedupes by ``command_id`` (the journal id) in ``write_commits``,
and the journal is keyed by ``(profile_id, idempotency_key)``. A crash or stop
with entries still queued loses nothing: they stay ``prepared``/``dispatched``
in the journal and ``replay_pending`` commits them at the next start.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from superlocalmemory.storage.admission_journal import (
    AdmissionEntry,
    AdmissionJournal,
    AdmissionPayloadError,
    RememberRequest,
    TerminalAdmissionError,
)

logger = logging.getLogger("superlocalmemory.core.deferred_admission")

CommitFn = Callable[[AdmissionEntry, RememberRequest], Mapping[str, Any]]

_FIRST_BACKOFF_SECONDS = 0.05
_MAX_BACKOFF_SECONDS = 2.0
_STOP_JOIN_SECONDS = 10.0


class DeferredCommitter:
    """A single ordered worker that commits journal entries left by contention."""

    def __init__(self, journal: AdmissionJournal, commit: CommitFn) -> None:
        self._journal = journal
        self._commit = commit
        self._cond = threading.Condition()
        self._queue: OrderedDict[str, int] = OrderedDict()  # journal_id -> attempts
        self._inflight: str | None = None
        self._stopping = False
        self._thread: threading.Thread | None = None

    @property
    def stopped(self) -> bool:
        with self._cond:
            return self._stopping

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._queue) + (1 if self._inflight else 0)

    def defer(self, entry: AdmissionEntry) -> None:
        """Queue one durable journal entry for commit (idempotent per entry)."""
        with self._cond:
            if self._stopping:
                # Still durable: the next start replays it from the journal.
                return
            if entry.journal_id != self._inflight:
                self._queue.setdefault(entry.journal_id, 0)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="slm-deferred-remember", daemon=True,
                )
                self._thread.start()
            self._cond.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        """Block until every deferred entry is resolved; False on timeout."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._queue and self._inflight is None, timeout=timeout,
            )

    def stop(self) -> None:
        """Stop after the in-flight entry; queued entries stay in the journal."""
        with self._cond:
            self._stopping = True
            thread = self._thread
            self._cond.notify_all()
        if thread is not None:
            thread.join(timeout=_STOP_JOIN_SECONDS)
        with self._cond:
            if self._queue:
                logger.info(
                    "%d accepted remember(s) not yet committed at stop; "
                    "they are durable and will be committed at the next start",
                    len(self._queue),
                )

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                journal_id, attempts = self._queue.popitem(last=False)
                self._inflight = journal_id
            try:
                done = self._finish(journal_id)
            except Exception as exc:  # e.g. the journal itself was busy
                logger.warning(
                    "deferred remember bookkeeping failed (%s); retrying",
                    type(exc).__name__,
                )
                done = False
            with self._cond:
                self._inflight = None
                if not done and not self._stopping:
                    # Back of the line, so one stuck entry cannot starve others.
                    self._queue[journal_id] = attempts + 1
                    delay = min(
                        _MAX_BACKOFF_SECONDS, _FIRST_BACKOFF_SECONDS * (2 ** attempts),
                    )
                    self._cond.notify_all()
                    self._cond.wait(timeout=delay)
                self._cond.notify_all()

    def _finish(self, journal_id: str) -> bool:
        """Commit one entry. True when resolved for good, False to retry."""
        try:
            entry = self._journal.get(journal_id)
            if entry.state in {"committed", "rejected"}:
                return True
            receipt = self._commit(entry, self._journal.request_for(entry))
            self._journal.mark_committed(journal_id, receipt)
            return True
        except TerminalAdmissionError as exc:
            self._journal.mark_rejected(journal_id, exc.error_code)
            logger.warning(
                "an accepted remember was rejected by deterministic policy (%s)",
                exc.error_code,
            )
            return True
        except AdmissionPayloadError:
            # Retrying cannot help; the entry stays in the journal for an
            # operator, and the failure is loud rather than silent.
            logger.error(
                "an accepted remember cannot be decrypted by this machine's key; "
                "it remains in the admission journal (%s)", journal_id,
            )
            return True
        except Exception as exc:  # contention, a stalled writer, I/O
            logger.warning(
                "an accepted remember is not committed yet (%s); retrying",
                type(exc).__name__,
            )
            return False
