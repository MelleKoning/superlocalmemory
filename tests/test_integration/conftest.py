# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fixtures shared by the integration tests."""

from __future__ import annotations

import pytest

from tests._engine_closer import closing_retrieval_engines


@pytest.fixture(autouse=True)
def closes_retrieval_engines(monkeypatch):
    """Close every ``RetrievalEngine`` built by any test in this folder.

    Integration tests build engines inline (e.g. the Hopfield wiring tests)
    and never close them, leaving ``slm-recall-channel`` workers running.
    See ``tests/_engine_closer.py``.
    """
    yield from closing_retrieval_engines(monkeypatch)
