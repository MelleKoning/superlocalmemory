# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A save racing the first kind-column probe still sees the kind columns.

``has_memory_kind_columns`` caches "absent" for 5 s. It used to stamp that
cache BEFORE probing, so any other thread asking while the first probe was
still running was told the columns were absent - and a save in that window
was written without its kind, a caller's declared kind included. Found when
the start-up kind check moved to its own thread and probed at engine start.
"""

from __future__ import annotations

import threading

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager


def test_a_concurrent_caller_during_the_first_probe_sees_the_columns(tmp_path) -> None:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    probing, release = threading.Event(), threading.Event()
    real_execute = DatabaseManager.execute

    def slow_probe(self, sql, params=()):
        if "table_info(atomic_facts)" in sql and threading.current_thread().name == "first":
            probing.set()
            assert release.wait(timeout=10)
        return real_execute(self, sql, params)

    DatabaseManager.execute = slow_probe
    try:
        first = threading.Thread(target=db.has_memory_kind_columns, name="first")
        first.start()
        assert probing.wait(timeout=5)
        concurrent = db.has_memory_kind_columns()  # a save asking mid-probe
        release.set()
        first.join(timeout=10)
    finally:
        DatabaseManager.execute = real_execute
    assert concurrent is True
    assert db.has_memory_kind_columns() is True
