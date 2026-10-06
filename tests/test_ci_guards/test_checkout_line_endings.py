# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""CI guard: a Windows checkout holds the same bytes as every other checkout.

Git for Windows defaults to ``core.autocrlf=true`` and rewrote every text file
to CRLF on checkout. On the Windows CI runner that made the pinned benchmark
fixture fail its SHA-256 check, made the generated Copilot plugin look out of
date in 21 files, and would stop any shell script running under bash. The fix
is ``.gitattributes``; this checks Git actually applies it to the files that
broke, with ``core.autocrlf=true`` forced the way Windows sets it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="needs a git checkout of the repository",
)

MUST_STAY_LF = [
    "tests/test_benchmarks/fixtures/evo_memory_synthetic_v1.jsonl",
    "copilot-plugin/README.md",
    "copilot-plugin/.vscode/mcp.json",
    "plugin/scripts/slm-launch",
    "plugin/scripts/ensure-venv.sh",
    "src/superlocalmemory/ui/index.html",
]
PLATFORM_DEFAULT = ["bin/slm.bat", "plugin/scripts/slm-launch.bat"]


def _eol(path: str) -> str:
    done = subprocess.run(
        ["git", "-c", "core.autocrlf=true", "check-attr", "eol", "--", path],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return done.stdout.rsplit(":", 1)[1].strip()


@pytest.mark.parametrize("path", MUST_STAY_LF)
def test_text_files_are_checked_out_with_lf_everywhere(path):
    assert (ROOT / path).exists(), f"{path} moved; update this list"
    assert _eol(path) == "lf"


@pytest.mark.parametrize("path", PLATFORM_DEFAULT)
def test_batch_files_keep_the_platform_line_ending(path):
    assert (ROOT / path).exists(), f"{path} moved; update this list"
    assert _eol(path) in {"unset", "unspecified"}  # either way: core.eol decides
