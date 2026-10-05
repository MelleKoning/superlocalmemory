# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm doctor` finds a migration-error log beside a database that was moved.

The daemon writes the log next to the memory database it actually uses, and
SLM_DATA_DIR / SL_MEMORY_PATH / SLM_HOME can move that database away from the
canonical data folder. The search for that second folder named a module that
was never imported, raised NameError, and a blanket ``except`` swallowed it —
so the doctor reported "clean" while the log sat unread.
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.cli import commands


def test_a_log_beside_the_configured_database_is_found(tmp_path, monkeypatch) -> None:
    canonical = tmp_path / "canonical"
    moved = tmp_path / "moved"
    canonical.mkdir()
    moved.mkdir()
    log = moved / "migration-error-20261002.log"
    log.write_text("boom", encoding="utf-8")

    from superlocalmemory.core import config as config_mod
    from superlocalmemory.infra import data_root

    monkeypatch.setattr(data_root, "canonical_data_root", lambda: canonical)
    monkeypatch.setattr(config_mod.SLMConfig, "for_mode",
                        classmethod(lambda cls, mode: SimpleNamespace(
                            memory_db_path=None, db_path=moved / "memory.db")))

    assert commands._migration_error_logs() == [log]


def test_the_canonical_folder_is_still_searched(tmp_path, monkeypatch) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    log = canonical / "migration-error-1.log"
    log.write_text("boom", encoding="utf-8")

    from superlocalmemory.core import config as config_mod
    from superlocalmemory.infra import data_root

    monkeypatch.setattr(data_root, "canonical_data_root", lambda: canonical)
    monkeypatch.setattr(config_mod.SLMConfig, "for_mode",
                        classmethod(lambda cls, mode: SimpleNamespace(
                            memory_db_path=None, db_path=canonical / "memory.db")))

    assert commands._migration_error_logs() == [log]
