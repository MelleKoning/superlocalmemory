#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A reproducible Answer Check demo: real Laya, synthetic memories, no tricks.

    python scripts/demo_answer_check.py [--port 8799] [--data-dir DIR] [--keep] [--once]
                                        [--laya-home DIR]

Starts a separate SLM daemon on its own data folder, stores a dozen fictional
memories about "Project Kestrel", and asks two questions with the on-device
answer check (Laya): one the memories answer, and one they do not. It verifies
the second really comes back as "I don't have that" — every run, the same way,
inside the 3-second limit — and FAILS otherwise. The screenshot is real or the
script exits non-zero; there is no fake judge and no stubbed verdict.

Your own memories are never read or touched. ``--laya-home`` names the data
folder whose installed Laya to use (default: your own install); only its model
and interpreter are used, and only after they pass Laya's own check here.

Exit codes: 0 verified · 2 unsafe data folder · 3 Laya not ready · 4 could not
store the memories · 5 a verdict did not match · 6 the daemon did not start.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "answer_check_demo.json"
_ENV_ROOTS = ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")
NOT_READY = "Set up the on-device check first: Settings → Answer check → On this Mac."


class DemoError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


# -- safety ---------------------------------------------------------------------

@contextlib.contextmanager
def data_root(path: Path | None) -> Iterator[None]:
    """Point SLM at ``path`` for this block (None = no override at all)."""
    saved = {k: os.environ.pop(k) for k in _ENV_ROOTS if k in os.environ}
    if path is not None:
        os.environ["SLM_DATA_DIR"] = str(path)
    try:
        yield
    finally:
        for k in _ENV_ROOTS:
            os.environ.pop(k, None)
        os.environ.update(saved)


def owner_roots() -> list[Path]:
    """Every place the owner's real store could be: never a demo folder."""
    from superlocalmemory.infra.data_root import canonical_data_root

    with data_root(None):
        roots = {canonical_data_root().resolve(),
                 canonical_data_root(home=pwd.getpwuid(os.getuid()).pw_dir).resolve()}
    return sorted(roots)


def check_data_dir(requested: str | None, owners: list[Path]) -> tuple[Path, bool]:
    """The demo folder and whether this run created it. Raises DemoError(2)."""
    if not requested:
        return Path(tempfile.mkdtemp(prefix="slm-answer-check-demo-")).resolve(), True
    path = Path(requested).expanduser().resolve()
    for owner in owners:
        if path == owner or owner.is_relative_to(path) or path.is_relative_to(owner):
            raise DemoError(2, f"Refusing {path}: it is (or holds) your real memory store.")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise DemoError(2, f"Refusing {path}: it must be a new or empty folder.")
    path.mkdir(parents=True, exist_ok=True)
    return path, False


# -- Laya -----------------------------------------------------------------------

def find_laya(home: Path | None) -> Any:
    """Laya's status in ``home`` (default: the owner's), reading no memories."""
    from superlocalmemory.core import answer_check_state, laya_runtime
    from superlocalmemory.core.config import SLMConfig

    with data_root(home):
        from superlocalmemory.infra.data_root import canonical_data_root

        root = canonical_data_root()
        retrieval = answer_check_state.overlay(SLMConfig.load().retrieval, root)
        return laya_runtime.detect(retrieval)


def prepare_demo(demo: Path, laya: Any) -> None:
    """Mode A (no LLM, embeddings on this machine), and Laya checked here, from
    scratch, as the demo's answer check. Same answers on any machine: nothing
    depends on which local services happen to be running."""
    from superlocalmemory.core import answer_check_state, laya_runtime
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    with data_root(demo):
        SLMConfig.for_mode(Mode.A, base_dir=demo).save(mode_change=True)
        SLMConfig.write_current_mode("a", demo)
        status = laya_runtime.adopt(laya.python, laya.hf_home, laya.model_path)
        if status.state != laya_runtime.STATE_READY:
            raise DemoError(3, f"{NOT_READY} ({status.error or status.state})")
        answer_check_state.update(demo, lambda v: {
            **v, "sufficiency_judge": "laya", "sufficiency_python": status.python,
            "sufficiency_model": status.model_path, "sufficiency_hf_home": status.hf_home,
        }, seed=answer_check_state.field_defaults)


# -- talking to the demo daemon (loopback only) ----------------------------------------

def http(port: int, method: str, path: str, body: Any = None, *, token: str = "",
         timeout: float = 30.0) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Install-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:  # noqa: S310 — loopback
            return res.status, json.loads(res.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}


def wait_until(check, timeout_s: float, step_s: float = 1.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(step_s)
    return None


def start_daemon(demo: Path, port: int) -> subprocess.Popen:
    env = {**os.environ, "SLM_DATA_DIR": str(demo), "SLM_DAEMON_PORT": str(port)}
    for k in ("SL_MEMORY_PATH", "SLM_HOME"):
        env.pop(k, None)
    log = open(demo / "demo-daemon.log", "wb")  # noqa: SIM115 — closed with the process
    proc = subprocess.Popen([sys.executable, "-m", "superlocalmemory.server.unified_daemon",
                             "--start", f"--port={port}"], env=env, stdout=log, stderr=log)
    if not wait_until(lambda: http(port, "GET", "/health", timeout=3)[0] == 200, 120):
        proc.terminate()
        raise DemoError(6, f"The demo daemon did not start; see {demo / 'demo-daemon.log'}")
    return proc


def seed(demo: Path, port: int, memories: list[str]) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    with data_root(demo):
        token = ensure_install_token()
    for text in memories:
        status, _ = http(port, "POST", "/remember", {"content": text}, token=token)
        if status != 200:
            raise DemoError(4, f"Could not store a demo memory (HTTP {status}).")
    found = wait_until(lambda: (http(port, "POST", "/api/v3/recall/trace",
                                     {"query": "Project Kestrel", "limit": 20})[1]
                                .get("result_count", 0) >= 10), 90, 2.0)
    if not found:
        raise DemoError(4, "The demo memories did not become searchable in 90 s.")


def ask(port: int, question: str) -> dict[str, Any]:
    _, body = http(port, "POST", "/api/v3/recall/trace", {"query": question, "limit": 10})
    return body.get("answer_check") or {}


def warm(port: int, question: str) -> None:
    if not wait_until(lambda: ask(port, question).get("status") == "judged", 120, 2.0):
        raise DemoError(3, f"{NOT_READY} (Laya did not answer within 2 minutes)")


# -- verification (pure) ----------------------------------------------------------

def verify_runs(name: str, runs: list[dict[str, Any]], expect: dict[str, Any],
                ceiling_ms: float) -> list[str]:
    """Every run must match ``expect``, agree with the others, and fit the limit."""
    problems = []
    for i, run in enumerate(runs, 1):
        for key, want in expect.items():
            if run.get(key) != want:
                problems.append(f"{name} run {i}: {key} was {run.get(key)!r}, expected {want!r}")
        total = run.get("total_ms")
        if not isinstance(total, (int, float)) or total > ceiling_ms:
            problems.append(f"{name} run {i}: {total!r} ms is over the {ceiling_ms:.0f} ms limit")
    if len({(r.get("status"), r.get("abstained")) for r in runs}) > 1:
        problems.append(f"{name}: the runs disagree with each other")
    return problems


def table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [f"{'question':<46} {'status':<8} {'abstained':<9} {'confidence':>10} "
             f"{'needs':>6} {'total ms':>9}"]
    for q, r in rows:
        conf = r.get("answer_confidence")
        thr = r.get("threshold")
        lines.append(f"{q[:46]:<46} {str(r.get('status')):<8} {str(r.get('abstained')):<9} "
                     f"{conf if conf is not None else '-':>10} {thr if thr is not None else '-':>6} "
                     f"{r.get('total_ms') or '-':>9}")
    return "\n".join(lines)


def run_demo(args: argparse.Namespace) -> int:
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    demo, created = check_data_dir(args.data_dir, owner_roots())
    proc = None
    try:
        laya = find_laya(Path(args.laya_home).resolve() if args.laya_home else None)
        if getattr(laya, "state", "") != "ready":
            raise DemoError(3, NOT_READY)
        prepare_demo(demo, laya)
        proc = start_daemon(demo, args.port)
        seed(demo, args.port, fixture["memories"])
        questions = fixture["questions"]
        for spec in questions.values():   # the model loads, and the new memories settle
            warm(args.port, spec["q"])
        rows, problems = [], []
        for name, spec in questions.items():
            runs = [ask(args.port, spec["q"]) for _ in range(int(fixture["repeat"]))]
            rows += [(spec["q"], r) for r in runs]
            problems += verify_runs(name, runs, spec["expect"], float(fixture["ceiling_ms"]))
        print(table(rows))
        if problems:
            print("\nNOT VERIFIED:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 5
        print(f"\nVerified. Open http://127.0.0.1:{args.port}/#answercheck-pane and ask:")
        for spec in questions.values():
            print(f"  - {spec['q']}")
        if not args.once:
            print("Press Ctrl-C to stop the demo.")
            with contextlib.suppress(KeyboardInterrupt):
                while proc.poll() is None:
                    time.sleep(1)
        return 0
    except DemoError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=15)
        if created and not args.keep:
            shutil.rmtree(demo, ignore_errors=True)  # only a folder this run made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--laya-home", default=None)
    parser.add_argument("--fixture", default=str(FIXTURE))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run_demo(args)
    except DemoError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
