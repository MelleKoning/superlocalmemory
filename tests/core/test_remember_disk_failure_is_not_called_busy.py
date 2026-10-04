"""A save the disk refused is reported as a storage failure, not as overload."""

from __future__ import annotations

import sqlite3
import threading
from types import SimpleNamespace

import pytest

from superlocalmemory.core import remember_runtime as rr
from superlocalmemory.storage.journal_writer import (
    AdmissionJournalOverloaded,
    AdmissionJournalUnavailable,
)


def _runtime_raising(error: BaseException) -> rr.CanonicalRememberRuntime:
    def _remember(*_a, **_k):
        raise error

    runtime = rr.CanonicalRememberRuntime.__new__(rr.CanonicalRememberRuntime)
    runtime._started = True
    runtime._binding_lock = threading.Lock()
    runtime._generation = 1
    runtime._service = SimpleNamespace(remember=_remember)
    runtime._deferred = SimpleNamespace(defer=None)
    return runtime


def _request():
    return SimpleNamespace(profile_id="default", idempotency_key="k-disk")


def _disk_full() -> AdmissionJournalUnavailable:
    error = AdmissionJournalUnavailable("admission journal could not commit")
    error.__cause__ = sqlite3.OperationalError("database or disk is full")
    return error


def test_a_full_disk_is_reported_as_a_storage_failure() -> None:
    with pytest.raises(rr.CanonicalRememberUnavailable) as caught:
        _runtime_raising(_disk_full()).remember(_request(), actor=None)
    assert not isinstance(caught.value, rr.CanonicalRememberBusy)
    assert "too many saves" not in str(caught.value)
    assert "disk" in str(caught.value)


def test_a_full_queue_is_still_reported_as_busy() -> None:
    overloaded = AdmissionJournalOverloaded("queue full", retry_after_seconds=1)
    with pytest.raises(rr.CanonicalRememberBusy):
        _runtime_raising(overloaded).remember(_request(), actor=None)


def test_running_out_of_time_is_still_reported_as_busy() -> None:
    from superlocalmemory.storage.journal_writer import _BUSY_MESSAGE

    with pytest.raises(rr.CanonicalRememberBusy):
        _runtime_raising(AdmissionJournalUnavailable(_BUSY_MESSAGE)).remember(
            _request(), actor=None,
        )
