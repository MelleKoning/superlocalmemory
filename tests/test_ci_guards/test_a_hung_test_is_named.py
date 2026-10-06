# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A test that never finishes ends the run and is named, with its stack.

Windows CI parts were cancelled at the job limit with no log and no report,
so the stuck test could not be identified (tests/_hard_timeout_plugin.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests._portable import child_env_base

REPO_ROOT = Path(__file__).resolve().parents[2]

_STUCK = '''
import subprocess, sys, time

def test_finishes():
    pass

def test_never_finishes():
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    time.sleep(300)
'''


def _run(tmp_path: Path, source: str, *extra: str) -> subprocess.CompletedProcess:
    """A pytest run of its own, rooted in ``tmp_path``, with only this plugin.
    This run's data root would read as live to it, so it is not passed on."""
    (tmp_path / "test_stuck.py").write_text(source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")}
    env.update(PYTHONPATH=str(REPO_ROOT), **child_env_base(tmp_path / "home"))
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "tests._hard_timeout_plugin", "--rootdir", str(tmp_path),
         "-c", os.devnull, *extra, str(tmp_path / "test_stuck.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120,
    )


def test_a_stuck_test_ends_the_run_and_is_named_with_its_stack(tmp_path) -> None:
    report = tmp_path / "hung-test.txt"
    result = _run(tmp_path, _STUCK, "--hard-timeout", "3", "--hard-timeout-report", str(report))

    assert result.returncode == 3, result.stdout + result.stderr
    text = report.read_text(encoding="utf-8")
    assert "test_stuck.py::test_never_finishes" in text
    assert "test_finishes" not in text.split("\n", 1)[0]
    assert "in test_never_finishes" in text, text  # the stuck frame
    assert "time.sleep(300)" in text  # the child it started is listed
    assert "HARD TIMEOUT" in result.stderr


def test_a_test_inside_the_limit_is_untouched(tmp_path) -> None:
    result = _run(tmp_path, "import time\n\ndef test_ok():\n    time.sleep(1)\n",
                  "--hard-timeout", "30")
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_run_over_its_total_limit_names_the_test_it_was_on(tmp_path) -> None:
    report = tmp_path / "hung-test.txt"
    result = _run(tmp_path, _STUCK, "--hard-timeout-session", "3",
                  "--hard-timeout-report", str(report))

    assert result.returncode == 3, result.stdout + result.stderr
    first_line = report.read_text(encoding="utf-8").split("\n", 1)[0]
    assert "test_stuck.py::test_never_finishes" in first_line
    assert "whole run" in first_line
