# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A restore point is restored under Windows file rules too.

Windows refuses to delete or replace a file any handle still has open. On the
Windows runner the restore of a damaged store failed: deleting
``memory.db-wal`` raised WinError 32 because a connection to the live store
was still open. Here the same rule is enforced on any platform
(tests/_portable.emulate_windows_file_sharing).
"""

from __future__ import annotations

from tests._portable import emulate_windows_file_sharing
from tests.test_storage import test_restore_damaged_live as damaged_case


def test_a_request_made_before_the_damage_restores_under_windows_rules(
        tmp_path, monkeypatch) -> None:
    emulate_windows_file_sharing(monkeypatch)
    damaged_case.test_a_request_made_before_the_damage_still_restores_and_adds_back(tmp_path)


def test_a_damaged_store_restores_under_windows_rules(tmp_path, monkeypatch) -> None:
    emulate_windows_file_sharing(monkeypatch)
    damaged_case.test_preview_and_request_work_on_a_damaged_store(tmp_path)
