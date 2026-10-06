# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recycling the reranker worker never leaves recall unranked, or a worker orphaned.

Audit M-12: every 500 requests the worker was killed BEFORE its replacement
was warm, so recalls fell back to unranked order for as long as a cold load
takes (22-63 s in the daemon's logs). And the warm-up thread spawned a worker
without the lock that the request path holds when IT spawns one, so the two
could each start a worker and lose track of one.

These tests run a real subprocess -- a stand-in worker that speaks the same
JSON-lines protocol with a configurable load delay -- so process lifetime,
pipes and PIDs are real. No model is loaded.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from superlocalmemory.retrieval import reranker as mod
from superlocalmemory.retrieval.reranker import CrossEncoderReranker
from superlocalmemory.storage.models import AtomicFact

_FAKE_WORKER = textwrap.dedent('''
    import json, os, sys, time
    log = os.environ["FAKE_LOG"]
    with open(log, "a") as f:
        f.write(f"spawn {os.getpid()}\\n")
    for line in sys.stdin:
        req = json.loads(line)
        cmd = req.get("cmd")
        if cmd == "quit":
            break
        if cmd == "load":
            time.sleep(float(os.environ.get("FAKE_LOAD_DELAY", "0")))
            if os.path.exists(os.environ.get("FAKE_FAIL_FLAG", "/nonexistent")):
                print(json.dumps({"ok": False, "error": "could not load"}), flush=True)
                continue
            with open(log, "a") as f:
                f.write(f"loaded {os.getpid()}\\n")
            print(json.dumps({"ok": True, "backend": "fake"}), flush=True)
        elif cmd == "rerank":
            docs = req["documents"]
            print(json.dumps({"ok": True, "scores": [float(len(d)) for d in docs],
                              "pid": os.getpid()}), flush=True)
        elif cmd == "score":
            print(json.dumps({"ok": True, "score": 1.0}), flush=True)
        else:
            print(json.dumps({"ok": True}), flush=True)
''')


def _candidates() -> list[tuple[AtomicFact, float]]:
    return [
        (AtomicFact(fact_id="a", memory_id="m", content="x"), 0.9),
        (AtomicFact(fact_id="b", memory_id="m", content="xxx"), 0.5),
        (AtomicFact(fact_id="c", memory_id="m", content="xx"), 0.1),
    ]


def _alive(pid: int) -> bool:
    from superlocalmemory.core.platform_utils import is_pid_alive

    if not is_pid_alive(pid):  # never os.kill(pid, 0): that is Ctrl+C on Windows
        return False
    if os.name == "nt":
        return True  # no zombies: a Windows process is gone once it exits
    try:  # a zombie still answers kill(0); reap it if it is our child
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER, encoding="utf-8")
    log = tmp_path / "workers.log"
    log.write_text("", encoding="utf-8")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_FAIL_FLAG", str(tmp_path / "fail"))
    monkeypatch.setattr(mod, "_RERANKER_PID_FILE", tmp_path / ".reranker.pid")
    monkeypatch.setattr(
        CrossEncoderReranker, "_worker_argv",
        lambda self: [sys.executable, str(script)], raising=False,
    )
    built: list[CrossEncoderReranker] = []

    def make(load_delay: float = 0.0) -> CrossEncoderReranker:
        monkeypatch.setenv("FAKE_LOAD_DELAY", str(load_delay))
        rr = CrossEncoderReranker(model_name="fake", backend="")
        built.append(rr)
        return rr

    def spawned() -> list[int]:
        return [int(line.split()[1]) for line in log.read_text(encoding="utf-8").splitlines()
                if line.startswith("spawn")]

    make.spawned = spawned  # type: ignore[attr-defined]
    make.fail_flag = tmp_path / "fail"  # type: ignore[attr-defined]
    yield make
    for rr in built:
        rr.shutdown(timeout=2.0)
    for pid in spawned():
        if _alive(pid):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))  # TerminateProcess on Windows


def _warm(rr: CrossEncoderReranker) -> None:
    assert rr.warmup_sync(timeout=15), "the stand-in worker never became ready"


class TestRecycleIsBlueGreen:

    def test_recall_stays_ranked_while_the_replacement_warms(self, fake_worker) -> None:
        rr = fake_worker(load_delay=1.0)
        _warm(rr)
        old_pid = rr._worker_proc.pid
        rr._request_count = mod._WORKER_RECYCLE_AFTER

        statuses = []
        # Spawning the replacement plus its 1 s load took over 3 s on a loaded
        # machine; the loop ends at the swap, so a longer ceiling costs nothing
        # when healthy and observes more recalls when not.
        t_end = time.monotonic() + 20.0
        while time.monotonic() < t_end:
            _, applied, status = rr.rerank_with_status("q", _candidates())
            statuses.append(status)
            if rr._worker_proc is not None and rr._worker_proc.pid != old_pid:
                break
            time.sleep(0.02)
        assert set(statuses) == {"applied"}, (
            f"recalls went unranked during the recycle: {sorted(set(statuses))}"
        )
        assert rr._worker_proc.pid != old_pid, "the replacement was never swapped in"
        # and the new worker serves straight away
        results, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is True
        assert [f.fact_id for f, _ in results] == ["b", "c", "a"]

    def test_a_recall_arriving_during_the_swap_is_still_ranked(
        self, fake_worker, monkeypatch,
    ) -> None:
        """The swap holds the worker lock; a recall must wait it out, not skip.

        Recall uses a non-blocking acquire so concurrent recalls never queue
        behind each other's inference. The swap itself (pointer exchange, PID
        file, idle timer) took 3-80 ms under load, and a recall that landed in
        it fell back to unranked order -- the test above failed about half the
        time on 4.1.19 and 4.1.20 for exactly this. Widen the window
        deterministically and require the recall to be ranked.
        """
        rr = fake_worker(load_delay=0.2)
        _warm(rr)
        old_pid = rr._worker_proc.pid
        in_swap, release = threading.Event(), threading.Event()
        real_record = mod.CrossEncoderReranker._record_worker_pid

        def slow_record(pid: int) -> None:  # runs under the lock, in the swap
            if pid != old_pid:
                in_swap.set()
                release.wait(timeout=5)
            real_record(pid)

        monkeypatch.setattr(rr, "_record_worker_pid", slow_record)
        rr._request_count = mod._WORKER_RECYCLE_AFTER
        rr.rerank_with_status("q", _candidates())  # starts the replacement
        assert in_swap.wait(timeout=10), "the swap never started"
        threading.Timer(0.15, release.set).start()
        results, applied, status = rr.rerank_with_status("q", _candidates())
        assert status == "applied" and applied is True
        assert [f.fact_id for f, _ in results] == ["b", "c", "a"]
        assert rr._worker_proc.pid != old_pid

    def test_the_old_worker_is_retired_after_the_swap(self, fake_worker) -> None:
        rr = fake_worker(load_delay=0.2)
        _warm(rr)
        old_pid = rr._worker_proc.pid
        rr._request_count = mod._WORKER_RECYCLE_AFTER
        rr.rerank_with_status("q", _candidates())
        assert _wait_for(lambda: rr._worker_proc is not None
                         and rr._worker_proc.pid != old_pid)
        assert _wait_for(lambda: not _alive(old_pid)), "the old worker was orphaned"
        assert rr._request_count < 5
        assert int(mod._reranker_pid_file().read_text(encoding="utf-8")) == rr._worker_proc.pid

    def test_one_replacement_at_a_time(self, fake_worker) -> None:
        rr = fake_worker(load_delay=0.5)
        _warm(rr)
        rr._request_count = mod._WORKER_RECYCLE_AFTER
        for _ in range(20):
            rr.rerank_with_status("q", _candidates())
        assert _wait_for(lambda: not rr._replacing)
        assert len(fake_worker.spawned()) == 2, fake_worker.spawned()

    def test_a_replacement_that_fails_to_load_keeps_the_current_worker(
        self, fake_worker,
    ) -> None:
        rr = fake_worker(load_delay=0.1)
        _warm(rr)
        old_pid = rr._worker_proc.pid
        fake_worker.fail_flag.write_text("1", encoding="utf-8")
        rr._request_count = mod._WORKER_RECYCLE_AFTER
        rr.rerank_with_status("q", _candidates())
        assert _wait_for(lambda: not rr._replacing)
        assert rr._worker_proc.pid == old_pid
        _, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is True
        failed_new = [p for p in fake_worker.spawned() if p != old_pid]
        assert failed_new and _wait_for(lambda: not _alive(failed_new[0]))
        # it waits another full cycle before trying again, rather than every request
        assert rr._request_count < mod._WORKER_RECYCLE_AFTER

    def test_shutdown_during_a_replacement_leaves_no_worker_behind(
        self, fake_worker,
    ) -> None:
        rr = fake_worker(load_delay=3.0)
        _warm(rr)
        rr._request_count = mod._WORKER_RECYCLE_AFTER
        rr.rerank_with_status("q", _candidates())
        assert _wait_for(lambda: len(fake_worker.spawned()) == 2)
        rr.shutdown(timeout=2.0)
        for pid in fake_worker.spawned():
            assert _wait_for(lambda: not _alive(pid), timeout=5), f"{pid} survived"


class TestNoOrphanWorker:

    def test_warm_up_and_a_request_cannot_both_spawn_a_worker(self, fake_worker) -> None:
        """The warm-up thread spawned without the lock the request path holds.

        Widen the check-then-spawn window so both paths are inside it at once:
        with the race, two workers start and one is referenced by nothing.
        """
        real_alive = mod._is_reranker_worker_alive

        def slow_alive() -> bool:
            answer = real_alive()   # check ...
            time.sleep(0.3)         # ... then act late: the race window, widened
            return answer

        with patch.object(mod, "_is_reranker_worker_alive", slow_alive):
            rr = fake_worker(load_delay=0.0)  # starts the warm-up thread
            time.sleep(0.05)
            caller = threading.Thread(target=rr.score_pair, args=("q", "d"))
            caller.start()
            caller.join(timeout=10)
            _warm(rr)
        assert len(fake_worker.spawned()) == 1, (
            f"{len(fake_worker.spawned())} workers spawned for one reranker"
        )

    def test_two_cold_recalls_start_one_warm_up(self, fake_worker) -> None:
        with patch.object(CrossEncoderReranker, "_start_background_warmup"):
            rr = fake_worker(load_delay=0.3)
        barrier = threading.Barrier(8)

        def cold_recall() -> None:
            barrier.wait()
            rr.rerank_with_status("q", _candidates())

        threads = [threading.Thread(target=cold_recall) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        _warm(rr)
        assert len(fake_worker.spawned()) == 1
