# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The one real, offline, model-loading check for Laya's verify().

Every other laya_runtime test mocks the subprocess boundary so the suite
never loads a model. This test is the exception: it runs verify() for real,
against a Laya install already present on this machine, entirely offline
(HF_HUB_OFFLINE=1 is set inside verify() itself; nothing here reaches the
network). It proves the production worker protocol and the canary threshold
actually work end to end — not just against the fakes.

Point it at an install with SLM_TEST_LAYA_PYTHON (the interpreter that has
laya-mlx) and SLM_TEST_LAYA_HF_HOME (its model cache). Skipped (not failed)
when either is unset or absent, and excluded from the default run by the
`native` marker (see pyproject.toml addopts).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from superlocalmemory.core import laya_runtime as lr

# From the environment, never a real home path: this repository is published.
_PYTHON = Path(os.environ.get("SLM_TEST_LAYA_PYTHON", "/nonexistent/laya/python"))
_HF_HOME = Path(os.environ.get("SLM_TEST_LAYA_HF_HOME", "/nonexistent/laya/hf-cache"))
_MODEL_PATH = (
    _HF_HOME / "hub" / "models--aac6fef--laya-mlx" / "snapshots"
    / lr.LAYA_MODEL_REVISION
)

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        not (_PYTHON.exists() and _MODEL_PATH.is_dir()),
        reason="set SLM_TEST_LAYA_PYTHON and SLM_TEST_LAYA_HF_HOME to a local Laya install",
    ),
]


def test_verify_passes_against_the_real_local_install():
    ok, reason = lr.verify(str(_PYTHON), str(_HF_HOME), str(_MODEL_PATH), timeout_s=120.0)
    assert ok is True, reason
