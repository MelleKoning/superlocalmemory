# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Run a snippet under macOS Guard Malloc and report whether it wrote out of bounds.

Guard Malloc (``/usr/lib/libgmalloc.dylib``) places every allocation against
an unmapped page, so the first stray write faults at the instruction that made
it instead of corrupting the heap silently. The child turns that fault into an
ordinary exit code before it can become a macOS crash report, so a caught
regression fails the test without a crash dialog on the developer's screen.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

GUARD_MALLOC = Path("/usr/lib/libgmalloc.dylib")
GUARD_MALLOC_AVAILABLE = sys.platform == "darwin" and GUARD_MALLOC.exists()
SKIP_REASON = "Guard Malloc is the macOS allocator that faults on the first stray write"

_SRC = Path(__file__).resolve().parents[2] / "src"
_BANNER = "GuardMalloc["

# A fault exits with the signal number (SIGABRT 6, SIGBUS 10, SIGSEGV 11)
# through _exit, which is async-signal-safe and ends the process before a
# macOS crash report exists.
FAULT_TO_EXIT_CODE = textwrap.dedent(
    """
    import ctypes as _ctypes
    import signal as _signal
    _libc = _ctypes.CDLL(None)
    _libc.signal.restype = _ctypes.c_void_p
    _libc.signal.argtypes = [_ctypes.c_int, _ctypes.c_void_p]
    _exit_address = _ctypes.cast(_libc._exit, _ctypes.c_void_p).value
    for _fault in (_signal.SIGSEGV, _signal.SIGBUS, _signal.SIGABRT):
        _libc.signal(int(_fault), _exit_address)
    """
)


def run_under_guard_malloc(
    script: str, tmp_path: Path, *, timeout: float = 300,
) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a fresh interpreter with every allocation guarded."""
    env = {
        **os.environ,
        "DYLD_INSERT_LIBRARIES": str(GUARD_MALLOC),
        "PYTHONMALLOC": "malloc",
        "PYTHONPATH": str(_SRC),
        "HOME": str(tmp_path / "home"),
        "SLM_DATA_DIR": str(tmp_path / "data"),
    }
    return subprocess.run(
        [sys.executable, "-c", FAULT_TO_EXIT_CODE + textwrap.dedent(script)],
        env=env, capture_output=True, text=True, timeout=timeout,
    )


def assert_no_stray_write(done: subprocess.CompletedProcess[str], marker: str) -> None:
    """The child ran guarded, finished, and printed ``marker``."""
    assert _BANNER in done.stderr, "the child did not run under Guard Malloc"
    assert done.returncode == 0, (
        f"exit {done.returncode}: a stray memory write was caught\n"
        + "\n".join(line for line in done.stderr.splitlines()
                    if not line.startswith(_BANNER))[-2000:]
    )
    assert marker in done.stdout
