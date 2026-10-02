# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The reranker worker hands back the GPU memory each request leaves behind.

Measured on Apple Silicon with the real model and public text (LoCoMo
candidates, 250 rerank calls, documents of mixed length): the GPU allocator
kept the freed buffers of earlier requests and grew from one 1 GiB block to two
(1,126 MB -> 2,201 MB of graphics memory, 3.6 GB in all). Releasing the cache
once it grows past a headroom keeps it at one block, with byte-identical
scores; releasing after EVERY request did too, but made short requests about
1.7x slower, so it is not done when the cache has not grown.

These tests drive the real request loop with a stand-in model and a stand-in
``torch`` -- no model is loaded.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types

import pytest

import superlocalmemory.core.reranker_worker as worker


class _Model:
    def __init__(self, device_type: str) -> None:
        self.device = types.SimpleNamespace(type=device_type)

    def predict(self, pairs):
        return [float(len(doc)) for _query, doc in pairs]


MB = 1024 * 1024


class _Driver:
    """What the GPU driver, and the OS for the whole process, report holding."""

    def __init__(self) -> None:
        self.held = 1100 * MB
        self.footprint = 1600 * MB


def _fake_torch(monkeypatch, *, with_mps: bool = True, driver: _Driver | None = None,
                with_meter: bool = True) -> list[str]:
    calls: list[str] = []
    torch = types.ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext
    if with_mps:
        driver = driver or _Driver()
        mps = types.SimpleNamespace(empty_cache=lambda: calls.append("empty"))
        if with_meter:
            mps.driver_allocated_memory = lambda: driver.held
        torch.mps = mps
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(worker, "_baseline", {}, raising=False)
    monkeypatch.setattr(worker, "_phys_footprint",
                        lambda: driver.footprint if driver else None, raising=False)
    return calls


def _run(monkeypatch, model, requests: list[dict]) -> list[dict]:
    monkeypatch.setattr(worker, "_start_parent_watchdog", lambda: None)
    monkeypatch.setattr(worker, "_load_model", lambda name, backend: (model, "pytorch", name, ""))
    lines = "".join(json.dumps(r) + "\n" for r in requests + [{"cmd": "quit"}])
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(lines))
    monkeypatch.setattr(sys, "stdout", out)
    worker._worker_main()
    return [json.loads(line) for line in out.getvalue().splitlines()]


RERANK = {"cmd": "rerank", "query": "q", "documents": ["a", "bbb", "cc"]}
SCORE = {"cmd": "score", "query": "q", "document": "dddd"}


def test_a_cache_that_has_not_grown_is_kept_for_speed(monkeypatch) -> None:
    calls = _fake_torch(monkeypatch)
    replies = _run(monkeypatch, _Model("mps"), [{"cmd": "load"}, RERANK, RERANK, RERANK])
    assert [r["ok"] for r in replies] == [True, True, True, True]
    assert replies[1]["scores"] == [1.0, 3.0, 2.0], "scores must pass through untouched"
    assert calls == ["empty"], "only the post-load baseline release"


def test_a_cache_that_grew_past_the_headroom_is_handed_back(monkeypatch) -> None:
    driver = _Driver()

    class _Growing(_Model):
        def predict(self, pairs):
            driver.held += 1024 * MB      # the allocator reserved another block
            return super().predict(pairs)

    calls = _fake_torch(monkeypatch, driver=driver)
    replies = _run(monkeypatch, _Growing("mps"), [{"cmd": "load"}, RERANK, SCORE])
    assert replies[1]["scores"] == [1.0, 3.0, 2.0]
    assert replies[2] == {"ok": True, "score": 4.0}
    assert calls == ["empty", "empty", "empty"]


def test_growth_inside_the_headroom_is_left_alone(monkeypatch) -> None:
    driver = _Driver()

    class _Small(_Model):
        def predict(self, pairs):
            driver.held += worker.GPU_CACHE_HEADROOM_MB * MB // 4
            return super().predict(pairs)

    calls = _fake_torch(monkeypatch, driver=driver)
    _run(monkeypatch, _Small("mps"), [{"cmd": "load"}, RERANK, RERANK])
    # The baseline is taken after the load; each rerank then adds a quarter of
    # the headroom, so two of them stay inside it: nothing is released.
    assert calls == ["empty"]


def test_a_process_that_grew_past_its_headroom_is_trimmed(monkeypatch) -> None:
    """The allocator's cache also shows up outside graphics memory: measured,
    about 500 MB of it, which the release returns too."""
    driver = _Driver()

    class _Bloating(_Model):
        def predict(self, pairs):
            driver.footprint += (worker.FOOTPRINT_HEADROOM_MB + 1) * MB
            return super().predict(pairs)

    calls = _fake_torch(monkeypatch, driver=driver)
    _run(monkeypatch, _Bloating("mps"), [{"cmd": "load"}, RERANK])
    assert calls == ["empty", "empty"]


def test_memory_a_release_cannot_return_is_not_chased_every_request(monkeypatch) -> None:
    driver = _Driver()
    grown = {"n": 0}

    class _OnceThenFlat(_Model):
        def predict(self, pairs):
            if grown["n"] == 1:          # the first real request: grows, for good
                driver.footprint += 2 * worker.FOOTPRINT_HEADROOM_MB * MB
            grown["n"] += 1
            return super().predict(pairs)

    calls = _fake_torch(monkeypatch, driver=driver)
    _run(monkeypatch, _OnceThenFlat("mps"), [{"cmd": "load"}, RERANK, RERANK, RERANK])
    # the load, then one release for the growth -- not one per request after it
    assert calls == ["empty", "empty"]


def test_without_a_meter_it_releases_every_time(monkeypatch) -> None:
    calls = _fake_torch(monkeypatch, with_meter=False)
    _run(monkeypatch, _Model("mps"), [{"cmd": "load"}, RERANK, SCORE])
    assert len(calls) == 3, "memory first when growth cannot be measured"


def test_a_model_on_the_cpu_never_touches_the_gpu_cache(monkeypatch) -> None:
    calls = _fake_torch(monkeypatch)
    replies = _run(monkeypatch, _Model("cpu"), [{"cmd": "load"}, RERANK, SCORE])
    assert all(r["ok"] for r in replies)
    assert calls == []


@pytest.mark.parametrize("request_", [RERANK, SCORE])
def test_a_torch_without_the_gpu_module_still_answers(monkeypatch, request_) -> None:
    _fake_torch(monkeypatch, with_mps=False)
    replies = _run(monkeypatch, _Model("mps"), [{"cmd": "load"}, request_])
    assert replies[1]["ok"] is True


def test_a_failing_release_never_costs_an_answer(monkeypatch) -> None:
    torch = types.ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext

    def _boom() -> None:
        raise RuntimeError("driver said no")

    torch.mps = types.SimpleNamespace(empty_cache=_boom)
    monkeypatch.setitem(sys.modules, "torch", torch)
    replies = _run(monkeypatch, _Model("mps"), [{"cmd": "load"}, RERANK])
    assert replies[1] == {"ok": True, "scores": [1.0, 3.0, 2.0]}
