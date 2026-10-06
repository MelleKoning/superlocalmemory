# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Per-event evidence for every native hook SLM installs, Codex and Claude Code.

Handoff §13: "every native event in both hosts still requires its own
evidence: timestamp, event, exact command, deadline, exit status, sanitized
stdout/stderr". This test runs the EXACT command string each installer writes
(``codex_hooks.hook_definitions()`` and ``claude_code_hooks._hook_definitions``)
through a shell, with ``slm`` resolved to this source tree, and records one
evidence row per event. Each row must finish inside the host's own timeout
with output that host accepts.

What this is not: a host run. It proves the commands SLM writes behave inside
their deadlines; whether Codex / Claude Code fire them is the native-host step
of the release gate, recorded separately.

Set ``SLM_HOOK_EVIDENCE_OUT=<dir>`` to keep the JSON evidence files.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"

# Output each host accepts on exit 0, per its published hook spec.
#   Codex: plain text is context for SessionStart/UserPromptSubmit, ignored for
#   Pre/PostToolUse, invalid for Stop; no output is success everywhere.
#   Claude Code: plain text is context for SessionStart/UserPromptSubmit and a
#   transcript note elsewhere; JSON must parse when it is JSON.
_CODEX_TEXT_OK = {"SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse"}


def _shim(bin_dir: Path, *, mcp_mode: str = "real") -> None:
    """``slm`` on PATH = this source tree (optionally a hung/garbled ``slm mcp``)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "slm_shim.py"
    script.write_text(
        "import os, sys, time, runpy\n"
        f"mode = {mcp_mode!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'mcp' and mode != 'real':\n"
        "    pid_file = os.environ.get('SLM_SHIM_PID_FILE')\n"
        "    if pid_file:\n"
        "        open(pid_file, 'w').write(str(os.getpid()))\n"
        "    if mode == 'garbled':\n"
        "        sys.stdout.write('not json\\n{\"id\": 1, \"result\": \\n'); sys.stdout.flush()\n"
        "    time.sleep(300)\n"
        "    sys.exit(0)\n"
        f"sys.path.insert(0, {str(_SRC)!r})\n"
        "sys.argv = ['slm'] + sys.argv[1:]\n"
        "runpy.run_module('superlocalmemory.cli.main', run_name='__main__')\n",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        (bin_dir / "slm.cmd").write_text(
            f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8",
        )
    else:
        launcher = bin_dir / "slm"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8",
        )
        launcher.chmod(0o755)


@pytest.fixture
def host_env(tmp_path, monkeypatch):
    def build(mcp_mode: str = "real") -> dict[str, str]:
        bin_dir = tmp_path / f"bin-{mcp_mode}"
        _shim(bin_dir, mcp_mode=mcp_mode)
        env = dict(os.environ)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(_SRC)
        env["SLM_SHIM_PID_FILE"] = str(tmp_path / f"mcp-{mcp_mode}.pid")
        # Never a real daemon from a hook under test: the isolation plugin's
        # SLM_TEST_ISOLATION=1 stays set, and no spawn opt-in is passed on.
        env.pop("SLM_TEST_ALLOW_DAEMON_SPAWN", None)
        return env

    (tmp_path / "project").mkdir()
    return build


def _sanitise(text: str, tmp_root: Path) -> str:
    import tempfile

    for raw, label in ((str(tmp_root), "<TMP>"), (tempfile.gettempdir(), "<SYSTMP>"),
                       (os.path.realpath(tempfile.gettempdir()), "<SYSTMP>"),
                       (str(Path.home()), "<HOME>")):
        text = text.replace(raw, label)
    text = text.replace("/private<SYSTMP>", "<SYSTMP>")
    text = re.sub(r"[A-Za-z0-9_\-]{32,}", "<REDACTED>", text)
    return text[:300]


def _classify(stdout: str) -> str:
    body = stdout.strip()
    if not body:
        return "empty"
    try:
        value = json.loads(body)
    except ValueError:
        # Text that merely starts like JSON ("[SLM-AUTO] ...") is text; an
        # object that does not parse is a broken JSON reply.
        return "invalid_json" if body.startswith("{") else "text"
    return "json_object" if isinstance(value, dict) else "json_other"


def _run_event(host, event, command, deadline_s, env, tmp_path, payload) -> dict:
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    began = time.monotonic()
    try:
        proc = subprocess.run(
            command, shell=True, input=json.dumps(payload), capture_output=True,
            text=True, env=env, cwd=str(tmp_path / "project"), timeout=deadline_s + 10,
        )
        exit_code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code, out, err = None, exc.stdout or "", exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
    elapsed = time.monotonic() - began
    kind = _classify(out)
    if host == "codex":
        valid = kind in {"empty", "json_object"} or (kind == "text" and event in _CODEX_TEXT_OK)
    else:
        valid = kind != "invalid_json"
    return {
        "host": host, "event": event, "timestamp": started,
        "command": _sanitise(command, tmp_path), "deadline_s": deadline_s,
        "exit": exit_code, "elapsed_s": round(elapsed, 2), "within_deadline": elapsed < deadline_s,
        "stdout_kind": kind, "stdout_valid": valid,
        "stdout": _sanitise(out, tmp_path), "stderr": _sanitise(err, tmp_path),
    }


def _payload(event: str, tmp_path: Path) -> dict:
    return {
        "session_id": "evidence-session", "hook_event_name": event,
        "cwd": str(tmp_path / "project"), "prompt": "what did we decide about the probe",
        "tool_name": "Edit", "tool_input": {"file_path": str(tmp_path / "project" / "a.py")},
        "turn_id": "t1",
    }


def _keep(rows: list[dict], name: str, tmp_path: Path) -> None:
    out_dir = Path(os.environ.get("SLM_HOOK_EVIDENCE_OUT") or tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _assert_rows(rows: list[dict]) -> None:
    def _exit_ok(r):  # the opt-in gate blocks with exit 2 by design
        return r["exit"] == 0 or (r["exit"] == 2 and "session_init first" in r["command"])
    bad = [r for r in rows if not (r["within_deadline"] and r["stdout_valid"] and _exit_ok(r))]
    assert not bad, json.dumps(bad, indent=2)


def test_every_codex_native_hook_event_meets_its_deadline(host_env, tmp_path):
    from superlocalmemory.hooks.codex_hooks import hook_definitions

    env = host_env()
    rows = []
    for event, groups in hook_definitions().items():
        for group in groups:
            for hook in group["hooks"]:
                rows.append(_run_event("codex", event, hook["command"], hook["timeout"],
                                       env, tmp_path, _payload(event, tmp_path)))
    _keep(rows, "codex-native-hook-evidence.json", tmp_path)
    assert {r["event"] for r in rows} == {"SessionStart", "PostToolUse", "UserPromptSubmit", "Stop"}
    _assert_rows(rows)
    start = next(r for r in rows if r["event"] == "SessionStart")
    assert start["stdout_kind"] == "json_object"


def test_every_claude_code_native_hook_event_meets_its_deadline(host_env, tmp_path):
    from superlocalmemory.hooks.claude_code_hooks import _hook_definitions

    env = host_env()
    rows = []
    for event, groups in _hook_definitions(include_gate=True).items():
        for group in groups:
            for hook in group["hooks"]:
                rows.append(_run_event("claude_code", event, hook["command"], hook["timeout"],
                                       env, tmp_path, _payload(event, tmp_path)))
    _keep(rows, "claude-code-native-hook-evidence.json", tmp_path)
    assert {"SessionStart", "PostToolUse", "UserPromptSubmit", "Stop", "PreToolUse"} <= {
        r["event"] for r in rows}
    _assert_rows(rows)


def test_claude_code_hook_timeouts_are_seconds_not_milliseconds():
    """Claude Code reads ``timeout`` in seconds; 15000 meant about four hours."""
    from superlocalmemory.hooks.claude_code_hooks import _hook_definitions

    for groups in _hook_definitions(include_gate=True).values():
        for group in groups:
            for hook in group["hooks"]:
                assert 1 <= hook["timeout"] <= 30, hook


@pytest.mark.parametrize("mcp_mode", ["hang", "garbled"])
def test_codex_start_with_a_hung_or_garbled_mcp_still_answers_in_time(
    host_env, tmp_path, mcp_mode,
):
    from superlocalmemory.hooks.codex_hooks import hook_definitions

    env = host_env(mcp_mode)
    hook = hook_definitions()["SessionStart"][0]["hooks"][0]
    row = _run_event("codex", "SessionStart", hook["command"], hook["timeout"],
                     env, tmp_path, _payload("SessionStart", tmp_path))
    _keep([row], f"codex-start-{mcp_mode}-mcp.json", tmp_path)
    _assert_rows([row])
    assert row["stdout_kind"] == "json_object"
    assert "session_init unavailable" in row["stdout"]
    pid = int(Path(env["SLM_SHIM_PID_FILE"]).read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    assert not alive, "the hung slm mcp child outlived the hook"


def test_parallel_codex_and_claude_session_starts_both_meet_deadlines(host_env, tmp_path):
    from superlocalmemory.hooks.claude_code_hooks import _hook_definitions
    from superlocalmemory.hooks.codex_hooks import hook_definitions

    env = host_env()
    codex = hook_definitions()["SessionStart"][0]["hooks"][0]
    claude = _hook_definitions()["SessionStart"][1]["hooks"][0]
    rows: list[dict] = []
    jobs = [("codex", codex), ("claude_code", claude)]
    threads = [
        threading.Thread(target=lambda h=h, k=k: rows.append(_run_event(
            h, "SessionStart", k["command"], k["timeout"], env, tmp_path,
            _payload("SessionStart", tmp_path))))
        for h, k in jobs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    _keep(rows, "parallel-session-starts.json", tmp_path)
    assert len(rows) == 2
    _assert_rows(rows)
