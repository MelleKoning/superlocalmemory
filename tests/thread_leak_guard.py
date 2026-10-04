# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fail the test that leaves a product thread running behind it.

WHY THIS EXISTS
---------------
The full suite intermittently died with SIGSEGV in a numpy test while threads
started by much earlier tests were still alive: an engine's ``slm-sg-embed``
pool, the daemon's ``slm-enrich`` pool and LanceDB's background event loop.
None of those tests failed; the crash landed on whichever unrelated test was
running when the stray native work collided with it. A leak has to fail the
test that caused it, by name, or it is found only by bisecting a crash.

WHAT COUNTS
-----------
A thread that did not exist when the test started, is still alive after every
fixture of the test has been torn down (plus a short grace for threads that are
already on their way out), and has a product name: ``slm-*``/``slm_*``, or
``LanceDBBackgroundEventLoop``. LanceDB's loop belongs to the library and lives
for the rest of the process once ``lancedb`` is imported, so it is allowed in
tests marked ``native`` (the dedicated lane) and nowhere else.

A child process of the pytest process (an embedding or reranker worker, a
daemon a test spawned) that did not exist when the test started and is still
running at the module boundary counts the same way: it is reported by pid and
command line against the test that started it.

``SLM_THREAD_LEAK_REPORT=<path>`` appends JSON lines instead of failing, for an
inventory run over the whole suite.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

_PRODUCT_PREFIXES = ("slm-", "slm_")
_LANCEDB_LOOP = "LanceDBBackgroundEventLoop"
_GRACE_SECONDS = 1.0
_BEFORE = pytest.StashKey[frozenset]()
_BEFORE_PROCS = pytest.StashKey[frozenset]()
# Interpreter-lifetime helpers ``multiprocessing`` starts on first use and keeps
# for the rest of the process; they belong to the standard library, not a test.
_STDLIB_HELPERS = ("multiprocessing.resource_tracker", "multiprocessing.forkserver")


def _is_product_thread(thread: threading.Thread) -> bool:
    name = thread.name or ""
    return name.startswith(_PRODUCT_PREFIXES) or name == _LANCEDB_LOOP


def _leaked(before: frozenset, *, allow_lancedb: bool) -> list[threading.Thread]:
    return [
        t for t in threading.enumerate()
        if t.ident not in before
        and t.is_alive()
        and _is_product_thread(t)
        and not (allow_lancedb and t.name == _LANCEDB_LOOP)
    ]


def still_alive_after_grace(
    threads: list[threading.Thread], grace: float = _GRACE_SECONDS,
) -> list[threading.Thread]:
    """The given threads still alive after waiting up to ``grace`` seconds.

    The grace covers threads already on their way out (an executor shut down
    without waiting); it never waits on a thread that is not under suspicion.
    """
    deadline = time.monotonic() + grace
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return [t for t in threads if t.is_alive()]


def _children() -> dict[int, object]:
    """Running child processes of this process, by pid (empty without psutil)."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a runtime dependency
        return {}
    try:
        kids = psutil.Process().children(recursive=True)
    except psutil.Error:
        return {}
    alive = {}
    for proc in kids:
        try:
            if proc.status() == psutil.STATUS_ZOMBIE:
                continue
            cmdline = " ".join(proc.cmdline())
        except psutil.Error:
            continue
        if not any(helper in cmdline for helper in _STDLIB_HELPERS):
            alive[proc.pid] = proc
    return alive


def _describe(proc) -> str:
    try:
        cmd = " ".join(proc.cmdline())[:160]
    except Exception:  # noqa: BLE001 - the process may exit while we look
        cmd = "?"
    return f"process {proc.pid} ({cmd})"


def processes_alive_after_grace(procs: list, grace: float = _GRACE_SECONDS) -> list:
    """The given child processes still running after up to ``grace`` seconds."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return []
    if not procs:
        return []
    try:
        _gone, alive = psutil.wait_procs(procs, timeout=grace)
    except psutil.Error:
        return []
    return [p for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    item.stash[_BEFORE] = frozenset(t.ident for t in threading.enumerate())
    item.stash[_BEFORE_PROCS] = frozenset(_children())
    return (yield)


# Threads a test left behind, held until its module ends: a module- or
# class-scoped fixture may legitimately keep its engine across tests and is
# torn down only at the module boundary. ident -> (thread, originating test).
_PENDING: dict[int, tuple[threading.Thread, str]] = {}
# Same for child processes: pid -> (psutil.Process, originating test).
_PENDING_PROCS: dict[int, tuple[object, str]] = {}


def _module_of(item) -> str:
    return item.nodeid.split("::", 1)[0]


def _report_or_fail(item, leaked: list[tuple[str, str]]) -> None:
    by_origin: dict[str, list[str]] = {}
    for what, origin in leaked:
        by_origin.setdefault(origin, []).append(what)
    report = os.environ.get("SLM_THREAD_LEAK_REPORT")
    if report:
        with open(report, "a", encoding="utf-8") as fh:
            for origin, names in sorted(by_origin.items()):
                fh.write(json.dumps({"nodeid": origin, "threads": sorted(names)}) + "\n")
        return
    lines = [f"  {origin}: {', '.join(sorted(names))}" for origin, names in sorted(by_origin.items())]
    pytest.fail(
        "product thread(s) or child process(es) still running after the test "
        "that started them finished (close the engine/pool/backend/worker it "
        "created; see "
        "tests/thread_leak_guard.py):\n" + "\n".join(lines),
        pytrace=False,
    )


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_teardown(item, nextitem):
    result = yield
    before = item.stash.get(_BEFORE, None)
    if before is None:
        return result
    allow_lancedb = item.get_closest_marker("native") is not None
    for thread in _leaked(before, allow_lancedb=allow_lancedb):
        _PENDING.setdefault(thread.ident, (thread, item.nodeid))
    before_procs = item.stash.get(_BEFORE_PROCS, frozenset())
    for pid, proc in _children().items():
        if pid not in before_procs:
            _PENDING_PROCS.setdefault(pid, (proc, item.nodeid))
    if nextitem is not None and _module_of(nextitem) == _module_of(item):
        return result
    # Module boundary: every fixture of this module has been finalized.
    pending = list(_PENDING.values())
    pending_procs = list(_PENDING_PROCS.values())
    _PENDING.clear()
    _PENDING_PROCS.clear()
    if not pending and not pending_procs:
        return result
    alive = {t.ident for t in still_alive_after_grace([t for t, _ in pending])}
    leaked = [
        (f"{t.name}{' (daemon)' if t.daemon else ''}", origin)
        for t, origin in pending if t.ident in alive
    ]
    alive_pids = {p.pid for p in processes_alive_after_grace([p for p, _ in pending_procs])}
    leaked += [(_describe(p), origin) for p, origin in pending_procs if p.pid in alive_pids]
    if leaked:
        _report_or_fail(item, leaked)
    return result
