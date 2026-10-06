# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A model download started by the daemon does not outlive the daemon.

The daemon's self-heal starts the embedding and reranker downloads (up to ten
minutes each) as child processes. When the daemon stopped, the download kept
running on its own; on Windows nothing reaps it, and the Windows CI runner
caught it still alive in a later test.
"""

from __future__ import annotations

import subprocess
import sys
import time

from superlocalmemory.cli import setup_wizard
from superlocalmemory.core.platform_utils import is_pid_alive

# The parent starts a child that runs the download prelude and then just waits
# (standing in for a long download), waits until the child is past the prelude
# (it creates the marker file), prints the child's PID, and exits.
_PARENT = (
    "import os, subprocess, sys, time; "
    "marker = sys.argv[2]; "
    "child = subprocess.Popen([sys.executable, '-c', sys.argv[1] + "
    "'open(sys.argv[1], \"w\").close(); import time; time.sleep(120)', marker], "
    "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
    "deadline = time.monotonic() + 30; "
    "exec('while not os.path.exists(marker) and time.monotonic() < deadline: time.sleep(0.05)'); "
    "print(child.pid, flush=True)"
)


def _wait_until_gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.25)
    return not is_pid_alive(pid)


def test_the_download_child_exits_when_its_parent_is_gone(tmp_path) -> None:
    marker = tmp_path / "child-started"
    parent = subprocess.run(
        [sys.executable, "-c", _PARENT, setup_wizard._EXIT_WITH_PARENT, str(marker)],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
    )
    child_pid = int(parent.stdout.strip())
    try:
        assert marker.exists(), "the stand-in download never started"
        assert _wait_until_gone(child_pid, 20), (
            f"download child {child_pid} still running after its parent exited"
        )
    finally:
        if is_pid_alive(child_pid):
            from superlocalmemory.core.platform_utils import kill_process

            kill_process(child_pid)


def test_both_downloads_start_with_the_exit_with_parent_prelude() -> None:
    import inspect

    for download in (setup_wizard._download_model, setup_wizard._download_reranker):
        assert "_EXIT_WITH_PARENT" in inspect.getsource(download), download.__name__
