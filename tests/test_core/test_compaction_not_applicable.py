# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A store with no vector history to compact is not a failing maintenance step.

On a store whose vectors live in SQLite (the default until the vector index is
promoted), every maintenance run logged "vector store compaction skipped:
backend cannot compact" as a WARNING, counted it as a failed step, and after
three cycles escalated it to an ERROR saying the work "is not being done" --
for work that does not exist on that store.
"""

from __future__ import annotations

import logging

import pytest

from superlocalmemory.core import maintenance_scheduler as ms
from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler


class _Orchestrator:
    def __init__(self, backend) -> None:
        self._backend = backend

    def get_vector_backend(self):
        return self._backend


class _Plain:
    """A vector store with no version history (no ``compact``)."""


class _Recorder:
    _ESCALATE_AFTER = MaintenanceScheduler._ESCALATE_AFTER
    _record_step = MaintenanceScheduler._record_step
    failing_steps = MaintenanceScheduler.failing_steps
    _initial_vector_compaction = MaintenanceScheduler._initial_vector_compaction
    _running = True


@pytest.fixture(autouse=True)
def _fresh_once_state(monkeypatch):
    monkeypatch.setattr(ms, "_NOT_APPLICABLE_REPORTED", set(), raising=False)


@pytest.mark.parametrize("backend", [None, _Plain()], ids=["no-vector-index", "no-history"])
def test_it_is_reported_as_not_applicable_not_failed(monkeypatch, caplog, backend) -> None:
    monkeypatch.setattr(
        "superlocalmemory.core.backend_orchestrator.get_orchestrator",
        lambda: _Orchestrator(backend),
    )
    with caplog.at_level(logging.DEBUG, logger=ms.logger.name):
        out = ms.compact_vector_store()
    assert out["ok"] is True
    assert out["applicable"] is False
    assert "not applicable" in out["reason"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_it_is_said_once_not_every_run(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "superlocalmemory.core.backend_orchestrator.get_orchestrator",
        lambda: _Orchestrator(None),
    )
    with caplog.at_level(logging.INFO, logger=ms.logger.name):
        for _ in range(5):
            ms.compact_vector_store()
    said = [r for r in caplog.records
            if r.levelno >= logging.INFO and "not applicable" in r.getMessage()]
    assert len(said) == 1


def test_the_scheduler_never_counts_it_as_a_failing_step(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "superlocalmemory.core.backend_orchestrator.get_orchestrator",
        lambda: _Orchestrator(_Plain()),
    )
    recorder = _Recorder()
    with caplog.at_level(logging.INFO):
        for _ in range(MaintenanceScheduler._ESCALATE_AFTER + 2):
            recorder._initial_vector_compaction()
    assert recorder.failing_steps() == {}
    assert "has failed" not in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_real_compaction_failure_is_still_a_failure(monkeypatch, caplog) -> None:
    class _Broken:
        def compact(self, **_kw):
            raise RuntimeError("lance is unhappy")

    monkeypatch.setattr(
        "superlocalmemory.core.backend_orchestrator.get_orchestrator",
        lambda: _Orchestrator(_Broken()),
    )
    recorder = _Recorder()
    with caplog.at_level(logging.WARNING):
        recorder._initial_vector_compaction()
    assert recorder.failing_steps() == {"vector compaction": 1}
    assert "unhappy" in caplog.text
