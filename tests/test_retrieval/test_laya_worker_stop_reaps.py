# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A Laya worker that is slow to die after a kill is still collected, never left a zombie.

Same rule as the recall workers (retrieval/_worker_process.py): a bounded wait that
gives up must hand the killed process to a background reaper (Muse audit 2026-10-08, E1).
"""

from __future__ import annotations

import subprocess
import sys
import time

from superlocalmemory.retrieval import laya_transport


def test_a_laya_worker_still_dying_after_kill_is_reaped() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    real_wait = proc.wait

    def slow_to_die(timeout=None):
        if timeout is not None:  # every bounded wait gives up, as on a wedged worker
            raise subprocess.TimeoutExpired(proc.args, timeout)
        return real_wait()

    proc.wait = slow_to_die  # type: ignore[method-assign]
    try:
        laya_transport.stop_process(proc, graceful=False, wait_s=0.1)  # never raises
        deadline = time.monotonic() + 15
        while proc.returncode is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.returncode is not None, "the killed Laya worker was left unreaped"
    finally:
        if proc.returncode is None:
            proc.kill()
            real_wait(timeout=10)
