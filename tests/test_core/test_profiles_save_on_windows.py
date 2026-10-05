# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Profiles can be created and saved on Windows.

``ProfileManager._save`` wrote ``profiles.json`` through a temporary file but
kept that file's descriptor open while renaming it into place. Windows refuses
to rename an open file (WinError 32), so on Windows a ProfileManager could not
even be constructed: every profile test failed in setup on the CI runner.
"""

from __future__ import annotations

import json

from superlocalmemory.core.profiles import ProfileManager
from tests._portable import emulate_windows_file_sharing


def test_a_new_store_writes_its_default_profile(tmp_path, monkeypatch):
    emulate_windows_file_sharing(monkeypatch)
    ProfileManager(tmp_path)
    saved = json.loads((tmp_path / "profiles.json").read_text(encoding="utf-8"))
    assert [p["name"] for p in saved["profiles"]] == ["default"]
    assert list(tmp_path.glob("*.tmp")) == [], "a temporary file was left behind"


def test_a_created_profile_is_saved(tmp_path, monkeypatch):
    emulate_windows_file_sharing(monkeypatch)
    manager = ProfileManager(tmp_path)
    manager.create_profile("work")
    reloaded = ProfileManager(tmp_path)
    assert reloaded.get_profile("work") is not None
