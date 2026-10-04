# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The test-isolation guard refuses live SLM state loudly, never silently.

The guard under test is exercised against a stand-in root in ``tmp_path``; the
real store is only ever checked for membership, never touched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import conftest
from tests.isolation_guard import LiveRootGuard, live_data_roots

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guard(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    return LiveRootGuard([live]), live


@pytest.mark.parametrize("event, args", [
    ("open", ("{live}/runtimes/laya/adopted.json", "w", 0)),
    ("os.mkdir", ("{live}/runtimes", 0o777, -1)),
    ("os.rename", ("{outside}/adopted.json.tmp", "{live}/adopted.json", -1, -1)),
    ("os.remove", ("{live}/memory.db", -1)),
    ("shutil.copyfile", ("{outside}/a", "{live}/b")),
    ("shutil.rmtree", ("{live}", None)),
    ("sqlite3.connect", ("{live}/memory.db",)),
])
def test_every_way_of_writing_into_a_live_root_is_refused_and_recorded(
    guard, tmp_path, event, args,
) -> None:
    g, live = guard
    filled = tuple(a.format(live=live, outside=tmp_path) if isinstance(a, str) else a
                   for a in args)
    with pytest.raises(PermissionError, match="live SLM state"):
        g.audit(event, filled)
    assert len(g.drain()) == 1
    assert g.drain() == []


def test_paths_outside_the_live_root_pass(guard, tmp_path) -> None:
    g, live = guard
    g.audit("open", (str(tmp_path / "elsewhere.json"), "w", 0))
    g.audit("os.mkdir", (str(tmp_path / "livex"), 0o777, -1))  # prefix, not child
    g.audit("sqlite3.connect", (":memory:",))
    g.audit("open", (3, "r", 0))  # a file descriptor carries no path
    assert g.drain() == []


def test_a_swallowed_refusal_is_still_on_record(guard) -> None:
    """Product state writers often catch OSError; the refusal must survive."""
    g, live = guard
    try:
        g.audit("open", (str(live / "daemon.pid"), "w", 0))
    except OSError:
        pass  # what a defensive writer does
    assert g.drain(), "a swallowed refusal vanished; the test would pass silently"


def test_the_account_store_is_protected_even_if_home_was_redirected() -> None:
    """Setting HOME or SLM_DATA_DIR before pytest must not unguard the real store."""
    fake_env = {"SLM_DATA_DIR": "/nonexistent/slm-elsewhere", "HOME": "/nonexistent/h"}
    roots = live_data_roots(fake_env)
    assert Path("/nonexistent/slm-elsewhere") in roots
    if sys.platform != "win32":
        import pwd
        account = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".superlocalmemory"
        assert account.resolve(strict=False) in roots


@pytest.mark.skipif(sys.platform == "win32", reason="pwd is POSIX-only")
def test_the_session_guard_is_installed_over_the_live_roots() -> None:
    import pwd
    account = (Path(pwd.getpwuid(os.getuid()).pw_dir) / ".superlocalmemory").resolve(
        strict=False)
    assert account in conftest._LIVE_ROOT_GUARD.protected
    assert conftest._LIVE_ROOT_GUARD.covers(account / "runtimes" / "laya" / "adopted.json")
    # and it is the active data root nowhere: every test runs elsewhere
    assert not conftest._LIVE_ROOT_GUARD.covers(os.environ["SLM_DATA_DIR"])


_RUNTIME_WRITER = """
import json
{fixture_import}
from superlocalmemory.core import laya_runtime as lr

def test_writes_runtime_state():
    run_dir = lr.runtime_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "adopted.json").write_text(json.dumps({{"verified": True}}))
"""


def _run_without_root_conftest(tmp_path: Path, *, pinned: bool) -> Path:
    home = tmp_path / ("home-pinned" if pinned else "home-unpinned")
    home.mkdir()
    test_file = tmp_path / ("test_pinned.py" if pinned else "test_unpinned.py")
    test_file.write_text(_RUNTIME_WRITER.format(fixture_import=(
        "from tests.isolation_guard import explicit_slm_root  # noqa: F401"
        if pinned else "")))
    env = {k: v for k, v in os.environ.items()
           if k not in ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")}
    env.update(HOME=str(home),
               PYTHONPATH=os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)]))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--noconftest", "--rootdir", str(tmp_path), str(test_file)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return home / ".superlocalmemory"


def test_runtime_writers_stay_in_tmp_without_the_root_conftest(tmp_path) -> None:
    """How adopted.json reached the real store on 2026-10-03: a test that
    writes runtime state, run where the root conftest is not loaded, resolves
    the default home root. The explicit_slm_root fixture keeps it in tmp_path.

    Run in a subprocess against a stand-in HOME, so the real store is never
    at risk; the unpinned control proves the check can see a leak.
    """
    leaked = _run_without_root_conftest(tmp_path, pinned=False)
    assert (leaked / "runtimes" / "laya" / "adopted.json").is_file(), (
        "control: without the fixture the write should land in the home store"
    )
    kept = _run_without_root_conftest(tmp_path, pinned=True)
    assert not kept.exists(), textwrap.shorten(str(list(kept.rglob("*"))), 500)
