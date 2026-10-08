# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A background computation's child process ends with the call that started it.

The graph-metrics pass and the vector-index decode run in a spawned child with a
timeout. A timed-out child used to keep running to completion with nobody
waiting for it (up to 900 s of CPU), and a child of a killed daemon ran on as an
orphan.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from superlocalmemory.core.background_process import run_in_child


def _alive_children(deadline_s: float) -> list:
    end = time.monotonic() + deadline_s
    while True:
        alive = multiprocessing.active_children()
        if not alive or time.monotonic() > end:
            return alive
        time.sleep(0.1)


def test_a_result_and_an_error_come_back() -> None:
    assert run_in_child(pow, 2, 10, timeout=60) == 1024
    with pytest.raises(ValueError):
        run_in_child(int, "not a number", timeout=60)
    assert _alive_children(3.0) == []


def test_a_timed_out_child_is_stopped() -> None:
    with pytest.raises(TimeoutError):
        run_in_child(time.sleep, 30, timeout=2)
    assert _alive_children(3.0) == []


@pytest.mark.skipif(os.name != "posix", reason="parent kill + getppid are POSIX")
def test_the_child_of_a_killed_parent_stops(tmp_path) -> None:
    pid_file = tmp_path / "child.pid"
    script = textwrap.dedent(f"""
        import multiprocessing, threading, time, pathlib
        from superlocalmemory.core.background_process import run_in_child
        if __name__ == "__main__":
            threading.Thread(target=run_in_child, args=(time.sleep, 60),
                             kwargs={{"timeout": 120}}, daemon=True).start()
            for _ in range(200):
                kids = multiprocessing.active_children()
                if kids:
                    pathlib.Path({str(pid_file)!r}).write_text(str(kids[0].pid))
                    break
                time.sleep(0.05)
            time.sleep(120)
    """)
    parent = subprocess.Popen([sys.executable, "-c", script], env=os.environ.copy())
    try:
        end = time.monotonic() + 30
        while not pid_file.exists() and time.monotonic() < end:
            time.sleep(0.1)
        child = int(pid_file.read_text())
        time.sleep(1.0)  # let the child finish starting
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=10)
        end = time.monotonic() + 8
        while time.monotonic() < end:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        os.kill(child, signal.SIGKILL)  # do not leave it behind
        pytest.fail("the child outlived its killed parent")
    finally:
        if parent.poll() is None:
            parent.kill()
