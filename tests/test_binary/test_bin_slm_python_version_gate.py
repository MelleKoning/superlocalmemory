# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``bin/slm`` and ``bin/slm.bat`` are the repository-clone launchers (the
npm-packaged runtime instead goes through its own .slm-venv, guaranteed
3.12+ by scripts/postinstall.js). Both ran whatever ``python3``/``python``
they found on PATH with no version check at all, so a contributor on
Debian 12's stock 3.11 (or macOS's stock 3.9) got cryptic failures deep
inside the CLI instead of one clear message up front (L3-22).

``bin/slm`` is tested by real execution (POSIX, runs on this host).
``bin/slm.bat`` is tested structurally — these tests run on macOS/Linux CI
and cannot execute a Windows batch file, matching the existing pattern in
tests/test_plugin/test_wf_windows_crossplatform.py.
"""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_SLM = REPO_ROOT / "bin" / "slm"
BIN_SLM_BAT = REPO_ROOT / "bin" / "slm.bat"


# ---------------------------------------------------------------------------
# bin/slm (POSIX) — real execution against a controllable fake interpreter
# ---------------------------------------------------------------------------

pytestmark_posix = pytest.mark.skipif(
    os.name == "nt", reason="bin/slm is POSIX; bin/slm.bat is tested structurally below",
)


def _fake_python_env(tmp_path: Path, fake_version: str) -> dict:
    """An env where bin/slm's only `python3` on PATH reports `fake_version`
    for `--version` and otherwise behaves like the dispatcher-fallback
    stub (prints a sentinel so we can tell the main CLI was invoked)."""
    fake_py = tmp_path / "fakepy.sh"
    fake_py.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        if [ "$1" = "--version" ]; then
            echo "Python {fake_version}"
            exit 0
        fi
        echo "PYFALLBACK $*"
        exit 0
    """))
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    bin_dir = tmp_path / "stub_bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(fake_py)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["SLM_HOOK_BINARY_DISABLED"] = "1"
    env["SLM_HOOK_BINARY"] = str(tmp_path / "nonexistent" / "slm-hook")
    return env


def _run_slm(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(BIN_SLM), "status"], env=env, capture_output=True, text=True, timeout=20,
    )


@pytest.mark.skipif(os.name == "nt", reason="bin/slm is POSIX")
@pytest.mark.parametrize("fake_version", ["3.9.6", "3.11.9"])
def test_bin_slm_refuses_an_unsupported_python(tmp_path, fake_version):
    env = _fake_python_env(tmp_path, fake_version)
    proc = _run_slm(env)
    assert proc.returncode != 0, (proc.stdout, proc.stderr)
    assert "3.12" in proc.stderr, (proc.stdout, proc.stderr)
    assert "PYFALLBACK" not in proc.stdout, "must not invoke the CLI on an unsupported interpreter"


@pytest.mark.skipif(os.name == "nt", reason="bin/slm is POSIX")
@pytest.mark.parametrize("fake_version", ["3.12.0", "3.13.1", "3.14.5"])
def test_bin_slm_accepts_a_supported_python(tmp_path, fake_version):
    env = _fake_python_env(tmp_path, fake_version)
    proc = _run_slm(env)
    assert "PYFALLBACK" in proc.stdout, (proc.stdout, proc.stderr)


# ---------------------------------------------------------------------------
# bin/slm.bat (Windows) — structural check, no execution
# ---------------------------------------------------------------------------

def test_slm_bat_exists():
    assert BIN_SLM_BAT.exists(), f"{BIN_SLM_BAT} not found"


def test_slm_bat_checks_the_python_major_minor_version():
    text = BIN_SLM_BAT.read_text(encoding="utf-8")
    # Must actually branch on a parsed major/minor pair, not merely print
    # "3.12+" in a message nobody enforces.
    assert "PY_MAJOR" in text and "PY_MINOR" in text, (
        "bin/slm.bat must parse the found interpreter's version and compare "
        "it against 3.12, not just find *a* python and run it"
    )
    assert "12" in text


def test_slm_bat_prints_a_clear_message_on_an_unsupported_python():
    text = BIN_SLM_BAT.read_text(encoding="utf-8")
    assert "3.12" in text
    assert "python.org" in text
