# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""#139 — can the host ever reach slm-launch.bat? (Windows only; runs on CI.)

plugin/.mcp.json names an EXTENSIONLESS command:
    "${CLAUDE_PLUGIN_ROOT}/scripts/slm-launch"
and slm-launch.bat used to say "Windows picks .bat automatically". It does not,
when the host spawns the command directly (no shell):

  * CreateProcess cannot start a batch file, and appends only ".exe" to a
    name without an extension (Win32 CreateProcess docs).
  * libuv, under Node's child_process.spawn without `shell`, tries only the
    literal name plus ".com" and ".exe" — "Since CreateProcess can start only
    .com and .exe files, only those extensions are tried" (libuv
    src/win/process.c). Node's docs: .bat/.cmd "cannot be launched" without a
    shell.

These tests turn that into evidence on a real Windows runner. A fixture copies
the launcher pair and swaps the .bat for one that only drops a marker file, so
"was the .bat reached?" has a yes/no answer:

  1. Python's subprocess (CreateProcess, no shell): .bat NOT reached.
  2. Node's spawn without a shell (the MCP stdio spawn path): .bat NOT reached.
  3. Control — through cmd.exe, the same path IS reached. Without this the
     first two could pass because the fixture is broken.

What this does NOT prove: which spawn the Claude Code binary uses today (it may
change). Node + libuv is the documented behaviour of a no-shell spawn, and
anthropics/claude-code#58510 reports the identical ENOENT for plugin MCP
servers whose command is a .cmd shim.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(platform.system() != "Windows", reason="Windows spawn semantics")

MARKER_BAT = '@echo off\r\necho reached> "%~dp0bat-was-reached.txt"\r\n'


@pytest.fixture()
def launcher(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(REPO / "plugin" / "scripts" / "slm-launch", scripts / "slm-launch")
    (scripts / "slm-launch.bat").write_text(MARKER_BAT, encoding="ascii", newline="")
    return scripts / "slm-launch"


def _reached(launcher: Path) -> bool:
    return (launcher.parent / "bat-was-reached.txt").exists()


def test_createprocess_does_not_reach_the_bat(launcher: Path) -> None:
    with pytest.raises(OSError):
        subprocess.run([str(launcher)], capture_output=True, timeout=30)
    assert not _reached(launcher)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_node_spawn_without_a_shell_does_not_reach_the_bat(launcher: Path) -> None:
    script = (
        "const r = require('child_process').spawnSync(process.argv[1], [], {shell: false});"
        "console.log(JSON.stringify({error: r.error ? r.error.code : null, status: r.status}));"
    )
    # Forward slashes, as Claude Code substitutes ${CLAUDE_PLUGIN_ROOT} on Windows.
    out = subprocess.run(
        ["node", "-e", script, launcher.as_posix()],
        capture_output=True, text=True, timeout=60, check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["error"] is not None, result
    assert not _reached(launcher)


def test_control_a_shell_does_reach_the_bat(launcher: Path) -> None:
    subprocess.run(["cmd.exe", "/d", "/c", str(launcher)], capture_output=True, timeout=30)
    assert _reached(launcher), "fixture broken: even cmd.exe did not reach the .bat"
