# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Two facts tests need that Windows answers differently from POSIX.

* ``posix_bash()`` — a bash that really is a POSIX shell. On Windows a bare
  ``"bash"`` handed to ``subprocess`` resolves through CreateProcess, which
  searches ``System32`` before ``PATH`` and finds the WSL launcher there: on a
  machine with no Linux distribution it prints a UTF-16 "Windows Subsystem for
  Linux has no installed distributions" and runs nothing.
* ``committed_executable(path)`` — whether a shipped script carries the
  executable bit. Windows files have no such bit (``st_mode`` is always
  ``0o666``), so there the bit that ships is the one Git recorded (mode
  ``100755``), which is also what a package build reads.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

#: Whether files here carry a POSIX executable bit. A module constant so a test
#: can exercise the Windows branch on any machine.
HAS_EXECUTABLE_BIT = os.name != "nt"


def _is_wsl_launcher(candidate: Path) -> bool:
    windows = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
    resolved = candidate.resolve()
    return resolved.is_relative_to(windows) or "WindowsApps" in resolved.parts


def posix_bash() -> str | None:
    """Absolute path of a POSIX bash, or None if this machine has none."""
    found = shutil.which("bash")
    if os.name != "nt":
        return found
    candidates = [Path(found)] if found else []
    git = shutil.which("git")
    if git:
        # Git for Windows: <root>\cmd\git.exe or <root>\mingw64\bin\git.exe.
        for root in Path(git).resolve().parents[:3]:
            candidates.append(root / "bin" / "bash.exe")
    for candidate in candidates:
        if candidate.is_file() and not _is_wsl_launcher(candidate):
            return str(candidate)
    return None


def require_posix_bash() -> str:
    bash = posix_bash()
    if bash is None:
        pytest.skip("no POSIX bash on this machine (on Windows, install Git for Windows)")
    return bash


def git_index_mode(path: Path) -> str | None:
    """The mode Git recorded for ``path`` (``"100755"``), or None if unknown."""
    if shutil.which("git") is None:
        return None
    done = subprocess.run(
        ["git", "ls-files", "--stage", "--", path.name],
        cwd=path.parent, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return done.stdout.split(maxsplit=1)[0]


def committed_executable(path: Path) -> bool:
    """Whether ``path`` carries the executable bit that ships with it."""
    if HAS_EXECUTABLE_BIT:
        return bool(path.stat().st_mode & stat.S_IXUSR)
    mode = git_index_mode(path)
    if mode is None:
        pytest.skip("Windows files have no executable bit and this is not a git checkout")
    return mode == "100755"


def open_paths() -> set[str]:
    """Real paths of the files this process has open right now."""
    import psutil

    return {os.path.realpath(entry.path) for entry in psutil.Process().open_files()}


def emulate_windows_file_sharing(monkeypatch) -> None:
    """Make renaming or deleting an open file fail, as it does on Windows.

    Python opens files on Windows without ``FILE_SHARE_DELETE``, so while any
    handle is open the file cannot be replaced, renamed over or deleted:
    ``PermissionError: [WinError 32] The process cannot access the file because
    it is being used by another process``. POSIX allows all three, which is how
    a missing ``close()`` before a rename passes everywhere else.
    """
    real_replace, real_rename, real_unlink = os.replace, os.rename, os.unlink

    def _refuse_if_open(*paths) -> None:
        busy = open_paths()
        for path in paths:
            real = os.path.realpath(os.fspath(path))
            if real in busy:
                raise PermissionError(
                    13, "[WinError 32] The process cannot access the file because "
                    "it is being used by another process", os.fspath(path))

    def replace(src, dst, *args, **kwargs):
        if not args and not kwargs:
            _refuse_if_open(src, dst)
        return real_replace(src, dst, *args, **kwargs)

    def rename(src, dst, *args, **kwargs):
        if not args and not kwargs:
            _refuse_if_open(src, dst)
        return real_rename(src, dst, *args, **kwargs)

    def unlink(path, *args, **kwargs):
        if not args and not kwargs:
            _refuse_if_open(path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "remove", unlink)
