# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Tiny stand-ins for the embedding model process.

``EmbeddingService`` talks to its worker over one JSON line per request. These
children speak the same protocol without loading a model, so the service's
real spawn/wait/kill paths run in tests with no network and no torch.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

# Reads the request and never answers: a wedged or still-loading model.
SILENT_WORKER = "import sys, time\nsys.stdin.readline()\ntime.sleep(600)\n"


def answering_worker(delay_seconds: float = 0.0) -> str:
    """A worker that answers every embed after ``delay_seconds`` (a load)."""
    return (
        "import json, sys, time\n"
        f"time.sleep({float(delay_seconds)!r})\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    if req.get('cmd') == 'quit':\n"
        "        break\n"
        "    vecs = [[0.5] * int(req['dimension']) for _ in req['texts']]\n"
        "    print(json.dumps({'ok': True, 'vectors': vecs}), flush=True)\n"
    )


class Spawner:
    """Replaces ``subprocess`` inside the embeddings module only.

    ``codes`` are used in spawn order; the last one repeats.
    """

    def __init__(self, *codes: str, on_spawn=None) -> None:
        self.codes = list(codes) or [SILENT_WORKER]
        self.on_spawn = on_spawn
        self.procs: list[subprocess.Popen] = []

    def Popen(self, argv, *args, **kwargs):  # noqa: N802 - mirrors subprocess
        code = self.codes[min(len(self.procs), len(self.codes) - 1)]
        proc = subprocess.Popen([sys.executable, "-c", code], *args, **kwargs)
        self.procs.append(proc)
        if self.on_spawn is not None:
            self.on_spawn()
        return proc

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            Popen=self.Popen, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL,
        )

    def alive(self) -> list[int]:
        return [p.pid for p in self.procs if p.poll() is None]

    def reap(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass


def install(monkeypatch, *codes: str, on_spawn=None) -> Spawner:
    """Route the embeddings module's spawns to ``codes``; no pressure check."""
    from superlocalmemory.core import embeddings as emb_mod

    sp = Spawner(*codes, on_spawn=on_spawn)
    monkeypatch.setattr(emb_mod, "subprocess", sp.namespace())
    monkeypatch.setattr(
        emb_mod.EmbeddingService, "_check_memory_pressure",
        staticmethod(lambda: True),
    )
    return sp
