# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Close every ``RetrievalEngine`` a test builds, and join its workers.

Shared by the conftests of folders whose tests build engines inline
(``tests/test_retrieval``, ``tests/test_integration``). A ``RetrievalEngine``
owns a channel pool (``slm-recall-channel``) and a query-embed pool; the
daemon closes its engine on shutdown, and so does every test here, without
having to opt in. ``close()`` is final, so use this only in folders where no
engine outlives a single test.
"""

from __future__ import annotations

from collections.abc import Iterator

from superlocalmemory.core.thread_join import join_threads
from superlocalmemory.retrieval.engine import RetrievalEngine


def closing_retrieval_engines(monkeypatch) -> Iterator[list[RetrievalEngine]]:
    """Generator body for an autouse fixture: track, yield, then close all."""
    built: list[RetrievalEngine] = []
    original_init = RetrievalEngine.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(RetrievalEngine, "__init__", _tracking_init)
    yield built
    for engine in built:
        threads = engine.worker_threads()
        engine.close()
        join_threads(threads, owner="test retrieval engine")
