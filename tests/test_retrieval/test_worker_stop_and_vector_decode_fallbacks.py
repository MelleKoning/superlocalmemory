"""A stopped worker is always reaped, and a bad decoded matrix falls back.

* ``stop_process``: a worker that is still dying when the kill's wait times
  out used to be left unreaped (a zombie until the daemon exits).
* ``arrays_in_child``: a matrix whose length does not fit its ids used to raise
  out of the index build instead of building in place.
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from superlocalmemory.retrieval import _worker_process, vector_index_build


def test_a_worker_still_dying_after_kill_is_reaped() -> None:
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
    proc.stdin.close()  # the quit request fails: the kill path runs
    try:
        _worker_process.stop_process(proc, timeout=0.1)  # never raises
        deadline = time.monotonic() + 15
        # Watch the attribute only: polling would reap it on the test's behalf.
        while proc.returncode is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.returncode is not None, "the killed worker was left unreaped"
    finally:
        if proc.returncode is None:
            proc.kill()
            real_wait(timeout=10)


def test_a_matrix_that_does_not_fit_builds_in_place(tmp_path, monkeypatch) -> None:
    from superlocalmemory.core import background_process

    raw = np.zeros(3, dtype=np.float32).tobytes()  # 3 floats for 2 ids x dim 2
    monkeypatch.setattr(background_process, "run_in_child",
                        lambda *a, **k: (["a", "b"], raw))

    class _Db:
        db_path = tmp_path / "memory.db"

    assert vector_index_build.arrays_in_child(_Db(), "default", 2) is None
