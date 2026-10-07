# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A read connection a caller never closed is still closed, on any thread.

Dashboard routes take a legacy read connection and close it by hand; on an
error path (e.g. the 400 for an unknown kind) they skipped the close. The
connection then lived until garbage collection, which can run on another
thread, where SQLite refused the close ("SQLite objects created in a thread can
only be used in that same thread"): the connection was never closed.
"""

from __future__ import annotations

import gc
import sqlite3
import sys
import threading

import pytest


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    return path


def _open_elsewhere(make):
    """Open and use it on a worker thread, as a route does, and never close it."""
    box = {}

    def work() -> None:
        conn = make()
        conn.execute("SELECT 1 FROM t")
        box.update(conn=conn, raw=conn._connection)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    return box


@pytest.mark.parametrize("which", ["route", "lease"])
def test_an_unclosed_read_connection_closes_when_dropped_on_another_thread(db, which,
                                                                           monkeypatch):
    from superlocalmemory.server.routes.helpers import _RouteReadConnection
    from superlocalmemory.storage.read_connection import ReadConnectionFactory

    raised = []
    monkeypatch.setattr(sys, "unraisablehook", lambda u: raised.append(u.exc_value))
    make = (lambda: _RouteReadConnection(db)) if which == "route" \
        else (lambda: ReadConnectionFactory(db).open())
    box = _open_elsewhere(make)
    raw = box.pop("raw")
    box.clear()  # the last reference goes here, on the main thread
    gc.collect()
    assert not raised, raised
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        raw.execute("SELECT 1")
