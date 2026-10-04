# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Once an embedding service is shut down it stays shut (4.1.20 audit C-1).

``MemoryEngine.close()`` shuts its embedder down. An embed already waiting on
the worker used to sit out the full response timeout (180 s by default), then
treat the closed pipe as a crashed worker and spawn a fresh 1 GB model process
nobody would ever stop -- and because that wait ran on an executor thread, the
interpreter could not exit until it ended (~184 s measured).

The workers here are tiny Python children standing in for the model process:
one that never answers and one that answers after an optional delay. No model,
no network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.core import embeddings as emb_mod
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService
from tests.helpers import fake_embedding_worker as fake

REPO_SRC = str(Path(__file__).resolve().parents[2] / "src")


@pytest.fixture()
def spawner(monkeypatch):
    made: list[fake.Spawner] = []

    def install(*codes: str, on_spawn=None) -> fake.Spawner:
        sp = fake.install(monkeypatch, *codes, on_spawn=on_spawn)
        made.append(sp)
        return sp

    # Long enough that only a wake-up -- never the timeout -- can end a wait.
    monkeypatch.setattr(emb_mod, "_SUBPROCESS_RESPONSE_TIMEOUT", 60)
    yield install
    for sp in made:
        sp.reap()


def _service() -> EmbeddingService:
    return EmbeddingService(EmbeddingConfig(dimension=4))


def test_an_embed_in_flight_stops_at_shutdown_and_spawns_nothing(spawner) -> None:
    sp = spawner()
    svc = _service()
    out: dict = {}

    def _embed() -> None:
        out["v"] = svc.embed("in flight")
        out["at"] = time.monotonic()

    th = threading.Thread(target=_embed, name="in-flight-embed")
    th.start()
    deadline = time.monotonic() + 10
    while not sp.procs and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)  # the request is written and the embed is waiting
    assert len(sp.procs) == 1

    t_shutdown = time.monotonic()
    svc.shutdown(timeout=1.0)
    th.join(timeout=10)

    assert not th.is_alive(), "the in-flight embed kept waiting after shutdown"
    assert out["v"] is None
    assert out["at"] - t_shutdown < 1.5, "the in-flight embed sat out its timeout"
    assert len(sp.procs) == 1, "a worker was respawned after shutdown"
    assert sp.alive() == []


def test_an_embed_after_shutdown_returns_at_once_and_spawns_nothing(spawner) -> None:
    sp = spawner()
    svc = _service()
    svc.shutdown(timeout=0.1)

    t0 = time.monotonic()
    assert svc.embed("after") is None
    assert svc.embed_batch(["a", "b"]) == [None, None]
    assert time.monotonic() - t0 < 0.5
    assert sp.procs == []
    assert svc.is_closed is True


def test_a_shutdown_during_the_spawn_checks_spawns_nothing(spawner, monkeypatch) -> None:
    sp = spawner()
    svc = _service()

    def _check_then_close() -> bool:
        svc._closed = True  # shutdown() started while the check ran
        return True

    monkeypatch.setattr(svc, "_check_memory_pressure", _check_then_close)
    assert svc.embed("racing") is None
    assert sp.procs == []


def test_a_shutdown_while_the_child_starts_leaves_no_orphan(spawner) -> None:
    holder: dict = {}
    sp = spawner(on_spawn=lambda: setattr(holder["svc"], "_closed", True))
    svc = holder["svc"] = _service()

    assert svc.embed("racing") is None
    assert len(sp.procs) == 1
    deadline = time.monotonic() + 5
    while sp.alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sp.alive() == [], "the child spawned during shutdown was left running"
    assert svc._worker_proc is None


def test_a_ready_response_is_not_delayed_by_the_shutdown_poll() -> None:
    r, w = os.pipe()
    with os.fdopen(r, "r") as reader, os.fdopen(w, "w") as writer:
        writer.write('{"ok": true}\n')
        writer.flush()
        t0 = time.monotonic()
        line = EmbeddingService._readline_with_timeout(
            reader, 5.0, cancelled=lambda: False,
        )
    assert line == '{"ok": true}\n'
    assert time.monotonic() - t0 < 0.05


def test_interpreter_exit_is_not_held_by_an_embed_in_flight(tmp_path) -> None:
    """No close() at all: the process must still exit promptly."""
    script = textwrap.dedent(
        f"""
        import concurrent.futures, subprocess, sys, time
        from types import SimpleNamespace
        from superlocalmemory.core import embeddings as E
        from superlocalmemory.core.config import EmbeddingConfig

        procs = []
        def _popen(argv, *a, **k):
            p = subprocess.Popen([sys.executable, "-c", {fake.SILENT_WORKER!r}], *a, **k)
            procs.append(p)
            return p
        E.subprocess = SimpleNamespace(
            Popen=_popen, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL)
        E.EmbeddingService._check_memory_pressure = staticmethod(lambda: True)
        svc = E.EmbeddingService(EmbeddingConfig(dimension=4))
        pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="slm-sg-embed")
        pool.submit(svc.embed, "in flight at exit")
        while not procs:
            time.sleep(0.01)
        time.sleep(0.2)
        print("exiting", flush=True)
        """
    )
    env = {
        **os.environ,
        "PYTHONPATH": REPO_SRC,
        "SLM_EMBED_RESPONSE_TIMEOUT": "120",
        "SLM_DATA_DIR": str(tmp_path),
        "HOME": str(tmp_path),
    }
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True,
        text=True, timeout=60,
    )
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "exiting" in proc.stdout
    assert elapsed < 15, f"exit was held for {elapsed:.1f}s by the in-flight embed"
    assert "did not respond" not in proc.stderr
