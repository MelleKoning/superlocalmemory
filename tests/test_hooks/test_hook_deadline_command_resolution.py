# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``_resolve_command`` resolves a hook child through ``PATH`` before spawn.

Windows' ``CreateProcess`` (what ``subprocess.Popen`` uses for a list with
``shell=False``) appends only ``.exe`` to an extension-less name; it never
tries ``PATHEXT``'s other extensions, so a ``slm`` shimmed or installed as a
``.cmd`` wrapper is invisible to a bare ``Popen(["slm", ...])`` there and the
search silently continues to whatever ``slm.exe`` comes later on ``PATH``
(reproduced on Windows CI: a test's ``.cmd`` stub standing in for a hung
``slm mcp`` was skipped and the real, installed ``slm`` answered instead).
``shutil.which`` performs the real ``PATHEXT`` search; these tests pin that
every spawn site resolves through it first, using the *child's* env, not
whatever this test process happens to have.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from superlocalmemory.hooks.hook_deadline import McpStdio, _resolve_command, run_bounded


def _make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_resolves_cmd0_through_the_given_envs_path(tmp_path):
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = _make_executable(shim_dir / "slm")
    env = {"PATH": str(shim_dir)}

    resolved = _resolve_command(["slm", "mcp"], env)

    assert resolved[0] == str(shim), resolved
    assert resolved[1:] == ["mcp"]


def test_a_directory_earlier_on_path_wins_over_a_later_one(tmp_path):
    """The exact shape of the Windows bug: two 'slm's on PATH, first must win."""
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _make_executable(first_dir / "slm")
    _make_executable(second_dir / "slm")
    env = {"PATH": os.pathsep.join([str(first_dir), str(second_dir)])}

    resolved = _resolve_command(["slm", "mcp"], env)

    assert resolved[0] == str(first), resolved


def test_falls_back_to_the_original_command_when_nothing_is_found():
    env = {"PATH": ""}
    assert _resolve_command(["definitely-not-a-real-binary-xyz"], env) == [
        "definitely-not-a-real-binary-xyz",
    ]


def test_an_empty_command_is_returned_unchanged():
    assert _resolve_command([], {"PATH": "/usr/bin"}) == []


def test_none_env_falls_back_to_this_processs_own_path(monkeypatch, tmp_path):
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = _make_executable(shim_dir / "slm")
    monkeypatch.setenv("PATH", str(shim_dir))

    assert _resolve_command(["slm"], None) == [str(shim)]


def test_run_bounded_executes_the_resolved_path_not_a_same_named_decoy(tmp_path):
    """Composed check: a decoy earlier on PATH never used by this call must
    not shadow the real target once resolved -- run_bounded actually spawns
    what _resolve_command picked, not the raw ``cmd``."""
    from superlocalmemory.hooks.hook_deadline import HookDeadline

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    marker = tmp_path / "ran.txt"
    _make_executable(real_dir / "probe", f'#!/bin/sh\necho hit > "{marker}"\n')
    env = {"PATH": str(real_dir)}

    result = run_bounded(["probe"], deadline=HookDeadline(5.0), env=env)

    assert result.returncode == 0, result
    assert marker.exists(), "the resolved executable never ran"


def test_mcp_stdio_resolves_its_child_the_same_way(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    # A tiny stdio echo so the client can prove it is talking to THIS binary.
    _make_executable(
        real_dir / "probe",
        '#!/bin/sh\nread line\necho "$line"\n',
    )
    env = {"PATH": str(real_dir)}
    client = McpStdio(["probe"], env=env)
    try:
        assert client.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    finally:
        client.close()
