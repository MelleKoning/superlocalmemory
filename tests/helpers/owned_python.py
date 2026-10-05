# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A Python environment the test itself owns, for tests that start Laya.

SLM only runs a Laya interpreter that no other account can change
(core/laya_interpreter.py): the real file, its folder and the environment's
folders must be owned by this user or root and writable by no one else.

``sys.executable`` does not meet that rule on every machine, and the rule is
right to refuse it there. GitHub's hosted runners install Python into a tool
cache their images make writable by every account (``chmod -R 777 /opt`` and
``$AGENT_TOOLSDIRECTORY`` on Ubuntu), so a test that hands the rule
``sys.executable`` - directly or through an environment's ``bin/python`` link -
is refused there and passes only on a developer's own machine.

``owned_environment()`` builds ``<root>/bin/python`` with ``<root>/pyvenv.cfg``,
every folder and file owned by this user with explicit owner-only write
permissions, so the real rule runs unchanged and accepts it on every POSIX
machine. ``bin/python`` is a small launcher that runs ``sys.executable``; the
rule checks who can change what SLM is handed, which is this launcher.

Windows is skipped, not faked: Laya ships for Apple silicon only, and the rule
reads owners and permission bits that Windows does not report (``os.stat``
gives every writable file mode 0o666), so SLM refuses every Laya interpreter
there by design.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest

#: Why a test that needs an owned interpreter does not run on Windows.
WINDOWS_REASON = ("Laya runs on Apple silicon only; its interpreter rule reads POSIX "
                  "owners and permission bits, which Windows does not report")


def owned_environment(root: Path) -> Path:
    """``<root>/bin/python``: an environment only this user can change."""
    if os.name == "nt":
        pytest.skip(WINDOWS_REASON)
    if not sys.executable:
        pytest.fail("sys.executable is empty: there is no interpreter to launch")
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    # Explicit modes: the result must not depend on the runner's umask.
    for folder in (root, bin_dir):
        folder.chmod(0o755)
    cfg = root / "pyvenv.cfg"
    cfg.write_text(f"home = {Path(sys.executable).parent}\n", encoding="utf-8")
    cfg.chmod(0o644)
    python = bin_dir / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
                      encoding="utf-8")
    python.chmod(0o755)
    return python


__all__ = ["WINDOWS_REASON", "owned_environment"]
