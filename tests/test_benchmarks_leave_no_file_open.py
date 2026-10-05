# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm benchmark`` and the learning-loop experiment close their databases.

Each runs in a temporary folder that is removed at the end. They left the
reward model's (and the bandit's) connection open, so on Windows removing the
folder failed with WinError 32 and the command errored after its work was done.
"""

from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from tests._portable import record_files_open_at_tempdir_cleanup

ROOT = Path(__file__).resolve().parents[1]


def test_slm_benchmark_closes_its_store(monkeypatch, capsys):
    from superlocalmemory.cli import escape_hatch

    # cmd_benchmark imports tempfile when it runs: check the module itself.
    still_open = record_files_open_at_tempdir_cleanup(monkeypatch, sys.modules[__name__])
    escape_hatch.cmd_benchmark(Namespace(json=True))
    assert still_open == [], still_open


def test_the_learning_loop_experiment_closes_its_stores(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmark"))
    import exp12_learning_loop_ablation as exp12

    still_open = record_files_open_at_tempdir_cleanup(monkeypatch, exp12)
    exp12.run(n_trials=2)
    assert still_open == [], still_open
