"""The memory check after every embed reads the counters in-process on macOS.

It runs inside the embedder's request lock after every embed, the recall's
query embedding included. It used to start a ``vm_stat`` process each time;
from a large multi-threaded daemon that cost 15 ms idle and seconds under load,
on the recall's own clock. The decision (free + inactive pages against the
floor) must stay the same.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import psutil
import pytest

from superlocalmemory.core import embeddings

GB = 1024 ** 3


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(embeddings.sys, "platform", "darwin")

    def no_process(*_a, **_k):
        raise AssertionError("the memory check must not start a process")

    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setenv("SLM_MIN_AVAILABLE_MEMORY_GB", "2.0")


def _vm(monkeypatch, free_gb: float, inactive_gb: float) -> None:
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(
        free=int(free_gb * GB), inactive=int(inactive_gb * GB),
        available=int(50 * GB), total=int(64 * GB)))


def test_low_free_plus_inactive_is_pressure(darwin, monkeypatch):
    _vm(monkeypatch, 0.5, 1.0)  # 1.5 GB < 2.0 GB floor
    assert embeddings.EmbeddingService._check_memory_pressure() is False


def test_enough_free_plus_inactive_is_safe(darwin, monkeypatch):
    _vm(monkeypatch, 1.0, 1.5)  # 2.5 GB >= floor; neither counter alone is enough
    assert embeddings.EmbeddingService._check_memory_pressure() is True


def test_inactive_pages_count_toward_headroom(darwin, monkeypatch):
    _vm(monkeypatch, 0.1, 3.0)
    assert embeddings.EmbeddingService._check_memory_pressure() is True
