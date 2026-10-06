# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Stop a test that never finishes, and say where it was stuck.

``-p tests._hard_timeout_plugin --hard-timeout 900 --hard-timeout-report hung-test.txt``

A test that blocks forever (a lock that is never released, a child process
that is never reaped, a socket that never answers) used to run the CI job
into its own time limit. The job was then cancelled and its log and report
were lost, so nobody could tell which test it was. ``faulthandler_timeout``
only prints a traceback and keeps waiting.

With this plugin, once one test (setup, call and teardown together) has run
for ``--hard-timeout`` seconds, the plugin writes the test's id, every
thread's stack, the processes the test run started and the files this process
holds open to the report file and to stderr, stops those processes, and ends
the run with exit status 3. The per-test report log written so far is intact,
so CI still lists every test that ran.

Runs on every platform; needs nothing beyond the standard library and psutil
(a SuperLocalMemory dependency).
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from pathlib import Path

import pytest

EXIT_STATUS = 3

# pytest captures output at the file-descriptor level once it starts, so a
# report written to fd 2 later would land in the capture, not the CI log.
# ``-p`` loads this module before capturing begins: keep the real stderr now.
try:
    _REAL_STDERR = os.fdopen(os.dup(2), "w", encoding="utf-8", errors="replace")
except OSError:
    _REAL_STDERR = None


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("hard-timeout")
    group.addoption("--hard-timeout", type=float, default=0.0, metavar="SECONDS",
                    help="end the run when one test runs this long (0 = off)")
    group.addoption("--hard-timeout-session", type=float, default=0.0, metavar="SECONDS",
                    help="end the run, naming the running test, after this long "
                         "in total (0 = off); keep it below the CI step limit")
    group.addoption("--hard-timeout-report", default="hung-test.txt", metavar="PATH",
                    help="where to write the stuck test's stacks and processes")


_running = {"nodeid": "(between tests)"}


def _describe_processes(out) -> None:
    try:
        import psutil
    except ImportError:
        out.write("psutil unavailable: no process listing\n")
        return
    me = psutil.Process()
    out.write(f"\n--- processes started by this test run (pid {me.pid}) ---\n")
    for child in me.children(recursive=True):
        try:
            command = " ".join(child.cmdline())[:300]
            out.write(f"pid {child.pid} ppid {child.ppid()} {child.status()}: {command}\n")
        except psutil.Error as exc:
            out.write(f"pid {child.pid}: unreadable ({type(exc).__name__})\n")
    out.write("\n--- files this process holds open ---\n")
    try:
        for opened in me.open_files():
            out.write(f"{opened.path}\n")
    except psutil.Error as exc:
        out.write(f"unreadable ({type(exc).__name__})\n")
    try:
        memory = psutil.virtual_memory()
        out.write(f"\nmemory available {memory.available // 2**20} MiB "
                  f"of {memory.total // 2**20} MiB\n")
    except Exception:  # noqa: BLE001 - diagnostics only
        pass


def _stop_children() -> None:
    try:
        import psutil
    except ImportError:
        return
    children = psutil.Process().children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(children, timeout=10)


def _write_report(out, nodeid: str, seconds: float) -> None:
    out.write(f"HARD TIMEOUT: {nodeid} still running after {seconds:.0f}s "
              f"({time.strftime('%Y-%m-%dT%H:%M:%S')})\n\n--- every thread ---\n")
    out.flush()
    faulthandler.dump_traceback(file=out, all_threads=True)
    _describe_processes(out)
    out.flush()


def _expire(nodeid: str, seconds: float, report: Path) -> None:
    try:
        with report.open("w", encoding="utf-8") as out:
            _write_report(out, nodeid, seconds)
    except OSError:
        pass
    stream = _REAL_STDERR or sys.__stderr__
    if stream is not None:
        _write_report(stream, nodeid, seconds)
        stream.write(f"\nEnding the test run (exit {EXIT_STATUS}); report: {report}\n")
        stream.flush()
    _stop_children()
    os._exit(EXIT_STATUS)


def _report_path(config: pytest.Config) -> Path:
    return Path(config.getoption("--hard-timeout-report")).resolve()


def pytest_sessionstart(session: pytest.Session) -> None:
    seconds = float(session.config.getoption("--hard-timeout-session") or 0)
    if seconds <= 0:
        return
    report = _report_path(session.config)

    def _session_expired() -> None:
        _expire(f"{_running['nodeid']} (whole run over its {seconds:.0f}s limit)",
                seconds, report)

    timer = threading.Timer(seconds, _session_expired)
    timer.daemon = True
    timer.name = "hard-timeout-session"
    timer.start()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem):
    _running["nodeid"] = item.nodeid
    seconds = float(item.config.getoption("--hard-timeout") or 0)
    if seconds <= 0:
        yield
        return
    report = _report_path(item.config)
    timer = threading.Timer(seconds, _expire, args=(item.nodeid, seconds, report))
    timer.daemon = True
    timer.name = "hard-timeout"
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
