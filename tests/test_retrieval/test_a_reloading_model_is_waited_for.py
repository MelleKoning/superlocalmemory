# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Only a model that has never loaded gets the bounded "warming" wait (R4).

4.1.20 stopped a recall from waiting on a model that is still loading for the
first time: the semantic channels report ``warming`` and the rest answer.
``EmbeddingService.is_warm`` is also False after the 30-minute idle unload and
after a memory-pressure kill, so every first question after a coffee break
lost its semantic channels too -- something 4.1.19 never did; it simply waited
for the reload. A model that loaded once is reloading, and is waited for.

Real ``EmbeddingService`` code paths; the model process is a tiny child that
speaks the worker protocol (tests/helpers/fake_embedding_worker.py).
"""

from __future__ import annotations

import time

import pytest

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService
from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval.query_embedding import QueryEmbedder
from tests.helpers import fake_embedding_worker as fake

_GUARD = 0.15     # stands in for CHANNEL_HANG_GUARD_SECONDS
_RELOAD = 0.8     # a reload that takes longer than the guard


@pytest.fixture()
def service_with(monkeypatch):
    made: list[tuple[fake.Spawner, EmbeddingService, QueryEmbedder]] = []

    def build(*codes: str):
        sp = fake.install(monkeypatch, *codes)
        svc = EmbeddingService(EmbeddingConfig(dimension=4))
        qe = QueryEmbedder(lambda: svc)
        made.append((sp, svc, qe))
        return sp, svc, qe

    yield build
    for sp, svc, qe in made:
        qe.close()
        svc.shutdown(timeout=1.0)
        sp.reap()


def test_a_model_that_never_loaded_is_still_bounded(service_with) -> None:
    """The 4.1.20 cold-start protection is unchanged."""
    _sp, svc, qe = service_with(fake.answering_worker(_RELOAD))
    assert svc.has_loaded_once is False

    t0 = time.monotonic()
    vector, status = qe.embed("first question", _GUARD)
    assert status == chstat.WARMING
    assert vector is None
    assert time.monotonic() - t0 < _RELOAD


def test_after_an_idle_unload_the_reload_is_waited_for(service_with) -> None:
    sp, svc, qe = service_with(fake.answering_worker(0.0), fake.answering_worker(_RELOAD))
    assert svc.embed("load it once") == [0.5] * 4
    assert svc.unload(timeout=1.0) is True   # the 30-minute idle timer's call
    assert svc.is_warm is False
    assert svc.has_loaded_once is True

    t0 = time.monotonic()
    vector, status = qe.embed("question after a break", _GUARD)
    assert (vector, status) == ([0.5] * 4, None)
    assert time.monotonic() - t0 >= _RELOAD * 0.9, "it did not wait for the reload"
    assert len(sp.procs) == 2


def test_after_a_memory_pressure_kill_the_reload_is_waited_for(
    service_with, monkeypatch,
) -> None:
    sp, svc, qe = service_with(fake.answering_worker(0.0), fake.answering_worker(_RELOAD))
    # Spawn check passes; the check after the answer reports pressure and
    # kills the worker; every later check passes again.
    pressure = iter([True, False])
    monkeypatch.setattr(
        svc, "_check_memory_pressure", lambda: next(pressure, True),
    )
    assert svc.embed("load it once") == [0.5] * 4
    assert svc.is_warm is False, "the pressure check should have killed the worker"
    assert svc.has_loaded_once is True

    vector, status = qe.embed("question under pressure", _GUARD)
    assert (vector, status) == ([0.5] * 4, None)
    assert len(sp.procs) == 2


def test_has_loaded_once_survives_unload_but_not_a_new_service(service_with) -> None:
    _sp, svc, _qe = service_with(fake.answering_worker(0.0))
    svc.embed("x")
    svc.unload(timeout=1.0)
    assert svc.has_loaded_once is True
    assert EmbeddingService(EmbeddingConfig(dimension=4)).has_loaded_once is False
