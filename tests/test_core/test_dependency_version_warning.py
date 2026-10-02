# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The dependency check states the supported version and the fix — nothing
it cannot back up. 5.3.0 was the supported pin until 4.1.18; the move was
for security advisories, not because the old version "blows up memory"."""

from __future__ import annotations

import warnings
from importlib import metadata

import superlocalmemory


def test_the_warning_names_the_supported_version_and_the_fix(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda dist: "5.3.0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        superlocalmemory._check_critical_deps()
    messages = [str(w.message) for w in caught]
    st = [m for m in messages if "sentence" in m]
    assert st, messages
    text = st[0]
    assert "blow-up" not in text and "blow up" not in text
    expected = superlocalmemory._REQUIRED_VERSIONS["sentence_transformers"]
    assert f"sentence-transformers=={expected}" in text
    assert "5.3.0" in text


def test_no_warning_when_the_supported_version_is_installed(monkeypatch):
    monkeypatch.setattr(
        metadata, "version",
        lambda dist: superlocalmemory._REQUIRED_VERSIONS[dist.replace("-", "_")],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        superlocalmemory._check_critical_deps()
    assert not caught
