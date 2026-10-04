# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""#139 — on native Windows the plugin's MCP server starts from the installed slm.

Windows only; runs on the Windows CI runner. The declared `.mcp.json` entry is
expanded the way Claude Code expands it (_mcp_expand.py) against the REAL
environment — so ComSpec is the runner's own — and then spawned WITHOUT a
shell, as the host spawns it:

  * through Python's subprocess (CreateProcess);
  * through Node's child_process.spawn with shell:false (libuv), when node is
    on PATH — the documented spawn path of a Node/Bun host.

`slm.exe` here is a real console launcher made by the same distlib
ScriptMaker pip uses for console scripts, so "resolves to slm.exe" is tested
against the file type pip and pipx actually install. PATH is pinned so the
runner's own pip-installed slm cannot be the one found.

What this does NOT prove: the exact spawn call inside the Claude Code binary
on a given release (it may change). The premise tests next door
(test_windows_mcp_spawn_premise.py) carry the same caveat.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_plugin._mcp_expand import expanded_argv

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(platform.system() != "Windows", reason="Windows spawn semantics")

ENTRIES = (
    ("Claude Code", REPO / "plugin" / ".mcp.json", REPO / "plugin"),
    ("Codex marketplace", REPO / "codex-plugin" / ".mcp.json", REPO / "codex-plugin"),
)
IDS = [e[0] for e in ENTRIES]

FAKE_SLM = '''\
import sys

def main():
    args = sys.argv[1:]
    if args[:2] == ["serve", "start"]:
        print("SERVE-ON-STDOUT")  # must be redirected away from MCP stdout
        return 0
    if args == ["mcp"]:
        print("FAKE-SLM-MCP " + sys.argv[0])
        return 0
    print("unexpected " + repr(args), file=sys.stderr)
    return 3
'''


def _make_slm_exe(bin_dir: Path, module_dir: Path) -> Path:
    from pip._vendor.distlib.scripts import ScriptMaker  # what pip uses

    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "fake_slm.py").write_text(FAKE_SLM, encoding="utf-8")
    bin_dir.mkdir(parents=True, exist_ok=True)
    maker = ScriptMaker(None, str(bin_dir))
    maker.executable = sys.executable
    maker.variants = {""}
    made = maker.make("slm = fake_slm:main")
    exe = bin_dir / "slm.exe"
    assert exe.exists(), f"ScriptMaker made {made}, no slm.exe"
    assert not any(p.suffix.lower() in {".bat", ".cmd"} for p in bin_dir.iterdir())
    return exe


def _env(path_dirs: list[Path], module_dir: Path) -> dict[str, str]:
    system_root = os.environ["SystemRoot"]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(p) for p in path_dirs] + [os.path.join(system_root, "System32"), system_root]
    )
    env["PYTHONPATH"] = str(module_dir)
    return env


def _spawn_python(argv: list[str], env: dict[str, str], cwd: Path):
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=cwd, timeout=60)


def _spawn_node(argv: list[str], env: dict[str, str], cwd: Path):
    script = (
        "const [cmd, ...args] = JSON.parse(process.argv[1]);"
        "const r = require('child_process').spawnSync(cmd, args, {shell: false, encoding: 'utf8'});"
        "process.stdout.write(JSON.stringify({error: r.error ? r.error.code : null,"
        " status: r.status, stdout: r.stdout, stderr: r.stderr}));"
    )
    node = shutil.which("node")
    out = subprocess.run([node, "-e", script, json.dumps(argv)], capture_output=True,
                         text=True, env=env, cwd=cwd, timeout=60, check=True)
    r = json.loads(out.stdout)
    assert r["error"] is None, r
    return subprocess.CompletedProcess(argv, r["status"], r["stdout"], r["stderr"])


SPAWNERS = [pytest.param(_spawn_python, id="createprocess")]
if shutil.which("node"):
    SPAWNERS.append(pytest.param(_spawn_node, id="node-no-shell"))


@pytest.mark.parametrize("spawn", SPAWNERS)
@pytest.mark.parametrize("name, mcp_json, root", ENTRIES, ids=IDS)
def test_declared_command_starts_the_installed_slm_exe(name, mcp_json, root, spawn, tmp_path) -> None:
    exe = _make_slm_exe(tmp_path / "bin", tmp_path / "mod")
    env = _env([tmp_path / "bin"], tmp_path / "mod")
    argv = expanded_argv(mcp_json, root.as_posix(), env)
    assert Path(argv[0]).name.lower() == "cmd.exe", argv

    where = subprocess.run(["where.exe", "slm"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert Path(where.stdout.splitlines()[0]).resolve() == exe.resolve(), where.stdout

    result = spawn(argv, env, tmp_path)
    assert result.returncode == 0, (result.stdout, result.stderr)
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("FAKE-SLM-MCP"), result.stdout
    assert "SERVE-ON-STDOUT" in result.stderr, "serve start output must go to stderr"


@pytest.mark.parametrize("spawn", SPAWNERS)
@pytest.mark.parametrize("name, mcp_json, root", ENTRIES, ids=IDS)
def test_missing_slm_gives_a_clear_message(name, mcp_json, root, spawn, tmp_path) -> None:
    (tmp_path / "mod").mkdir()
    env = _env([], tmp_path / "mod")
    assert subprocess.run(["where.exe", "/q", "slm"], env=env, cwd=tmp_path).returncode != 0
    argv = expanded_argv(mcp_json, root.as_posix(), env)

    result = spawn(argv, env, tmp_path)
    assert result.returncode != 0
    assert result.stdout.strip() == "", "nothing may reach the MCP stdout"
    assert "slm is not on PATH" in result.stderr, result.stderr
    assert "pipx install superlocalmemory" in result.stderr, result.stderr
