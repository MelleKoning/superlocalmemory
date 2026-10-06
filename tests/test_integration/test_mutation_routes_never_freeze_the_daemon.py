# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A memory change that waits for a busy store never freezes the daemon.

4.1.21 ran the dashboard's delete, forget, merge, scope and correction routes
on the daemon's single request loop. Each waits for the store's writer, so
while another writer held the store (a backfill batch, a materialization
checkpoint: seconds on a large store) every other request - /health, /recall,
every assistant's save - waited behind it. Measured on a copy of a 22k-fact
store: an edit froze all requests for 60 s.

Composed path only: a REAL daemon subprocess on a private port with its own
data folder; the store is held busy from outside by a plain SQLite write
transaction, exactly as another writer process would hold it.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import urllib.request
import uuid

import pytest

from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    REPO_ROOT,
    RealDaemon,
    _child_env,
    _foreign_daemon_pids,
    _reserve_private_port,
)

#: How long the store is held busy for each change.
HOLD_S = 8.0
#: A probe slower than this means the request loop was frozen behind the change.
FROZEN_S = 3.0
RUN = uuid.uuid4().hex[:6].upper()


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    import subprocess
    import sys

    root = tmp_path_factory.mktemp("no-freeze")
    data_root = root / "data"
    data_root.mkdir()
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    env = _child_env(data_root, port, root / "home", root / "cache")
    foreign = _foreign_daemon_pids()
    log = root / "daemon.log"
    with log.open("wb") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
             "--start", f"--port={port}"],
            stdout=handle, stderr=handle, env=env, cwd=str(REPO_ROOT),
            start_new_session=True,
        )
    real = RealDaemon(proc, port, data_root, log, env)
    try:
        real.wait_ready()
        real.wait_health_fast()
        # A first recall loads the search models; that cold start is not what
        # this file measures, so it is paid here, before any store is held.
        real.request("GET", "/recall", params={"q": "warm up", "limit": 1}, timeout=300)
        yield real
    finally:
        real.stop(foreign)


def _saved_fact(daemon: RealDaemon, label: str) -> str:
    token = f"NF{label}{RUN}"
    daemon.remember(f"Fixture note {token}: the synthetic gauge reads {len(label)} units.",
                    "", f"no-freeze-{label}-{RUN}")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        conn = sqlite3.connect(f"file:{daemon.data_root / 'memory.db'}?mode=ro", uri=True,
                               timeout=30)
        try:
            row = conn.execute("SELECT fact_id FROM atomic_facts WHERE content LIKE ? "
                               "ORDER BY rowid LIMIT 1", (f"%{token}%",)).fetchone()
        finally:
            conn.close()
        if row:
            return str(row[0])
        time.sleep(0.2)
    raise AssertionError(f"fact for {label} never became queryable")


class _StoreHeldBusy:
    """Another writer holding the store: BEGIN IMMEDIATE for ``seconds``."""

    def __init__(self, daemon: RealDaemon, seconds: float) -> None:
        self._path, self._seconds = daemon.data_root / "memory.db", seconds
        self._held = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        conn = sqlite3.connect(self._path, timeout=60, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._held.set()
            time.sleep(self._seconds)
            conn.execute("ROLLBACK")
        finally:
            conn.close()

    def __enter__(self):
        self._thread.start()
        assert self._held.wait(timeout=60)
        return self

    def __exit__(self, *exc) -> None:
        self._thread.join(timeout=self._seconds + 60)


def _probe_seconds(daemon: RealDaemon, path: str) -> float:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{daemon.port}{path}",
                                    timeout=HOLD_S * 3) as response:
            response.read()
    except Exception:  # an error answer is still an answer; only time matters
        pass
    return time.monotonic() - started


CHANGES = {
    "delete": lambda f, other: ("DELETE", f"/api/memories/{f}", None),
    "forget": lambda f, other: ("POST", f"/api/memories/{f}/forget", None),
    "merge": lambda f, other: ("POST", f"/api/memories/{f}/merge", {"into": other}),
    "scope": lambda f, other: ("PATCH", f"/api/memories/{f}/scope", {"scope": "personal"}),
    "edit": lambda f, other: ("PATCH", f"/api/memories/{f}",
                              {"content": f"Corrected fixture note {RUN}."}),
}


@pytest.mark.parametrize("change", sorted(CHANGES))
def test_a_change_waiting_on_a_busy_store_never_freezes_health_or_recall(
    daemon, change,
) -> None:
    fact = _saved_fact(daemon, f"{change}a")
    other = _saved_fact(daemon, f"{change}b")
    method, path, body = CHANGES[change](fact, other)
    answered: dict = {}

    with _StoreHeldBusy(daemon, HOLD_S):
        worker = threading.Thread(
            target=lambda: answered.update(code=daemon.request(
                method, path, body, timeout=HOLD_S * 6)[0]), daemon=True)
        worker.start()
        time.sleep(0.5)  # the change is now waiting for the store
        health = [_probe_seconds(daemon, "/health") for _ in range(3)]
        recall = _probe_seconds(daemon, f"/recall?q=gauge+{RUN}&limit=3")
    worker.join(timeout=HOLD_S * 6)

    assert "code" in answered, f"{change} never answered"
    assert max(health) < FROZEN_S, (change, health)
    assert recall < FROZEN_S, (change, recall)
