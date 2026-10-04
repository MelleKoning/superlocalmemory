# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fixtures shared by the retrieval tests."""

from __future__ import annotations

import pytest

from superlocalmemory.core.thread_join import join_threads
from superlocalmemory.retrieval.engine import RetrievalEngine


@pytest.fixture(autouse=True)
def closes_retrieval_engines(monkeypatch):
    """Close every ``RetrievalEngine`` the test builds, and join its workers.

    A ``RetrievalEngine`` owns a channel pool (``slm-recall-channel``) and a
    query-embed pool. Tests here build engines inline and through module
    helpers. Every engine built during any test in this folder is closed when
    the test ends, the way the daemon closes its engine on shutdown, so a new
    test cannot leak one by forgetting to opt in.
    """
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
