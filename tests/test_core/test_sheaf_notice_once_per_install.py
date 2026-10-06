# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The sheaf switch-off notice is given once per install, not once per command.

Each ``slm`` command is a new process, so a once-per-process guard printed the
notice on every command until something happened to save the config. Whether
it has been shown is now recorded beside config.json. The value it reports is
unchanged: an old stored default still reads as off on every load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from superlocalmemory.core import config_upgrades
from superlocalmemory.core.config import SLMConfig

OLD_DUMP = {
    "fisher_temperature": 15.0, "fisher_bayesian_update": True,
    "fisher_mode": "simplified", "langevin_dt": 0.005, "langevin_temperature": 0.3,
    "langevin_persist_positions": True, "langevin_weight_range": [0.0, 1.0],
    "ebbinghaus_langevin_coupling_enabled": False, "sheaf_at_encoding": True,
    "sheaf_contradiction_threshold": 0.45, "sheaf_max_edges_per_check": 200,
}


def _old_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "a", "math": OLD_DUMP}), encoding="utf-8")
    return path


def _load_as_a_new_process(path: Path, monkeypatch) -> SLMConfig:
    monkeypatch.setattr(config_upgrades, "_sheaf_switch_off_reported", False)
    return SLMConfig.load(path)


def _notices(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "math.sheaf_at_encoding" in r.getMessage()]


def test_three_commands_one_notice(tmp_path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="superlocalmemory.core.config")
    path = _old_config(tmp_path)

    loaded = [_load_as_a_new_process(path, monkeypatch) for _ in range(3)]

    assert len(_notices(caplog)) == 1
    # The behaviour the notice describes still holds on every load.
    assert all(c.math.sheaf_at_encoding is False for c in loaded)
    # And the user's file is not rewritten behind their back.
    assert json.loads(path.read_text(encoding="utf-8"))["math"] == OLD_DUMP


def test_a_second_install_gets_its_own_notice(tmp_path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="superlocalmemory.core.config")
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        _load_as_a_new_process(_old_config(tmp_path / name), monkeypatch)

    assert len(_notices(caplog)) == 2


def test_an_unwritable_folder_still_warns_once_per_process(
    tmp_path, monkeypatch, caplog,
) -> None:
    caplog.set_level(logging.WARNING, logger="superlocalmemory.core.config")
    path = _old_config(tmp_path)

    def _refuse(*_a, **_k):
        raise PermissionError("read-only")

    monkeypatch.setattr(config_upgrades.os, "open", _refuse)
    monkeypatch.setattr(config_upgrades, "_sheaf_switch_off_reported", False)
    SLMConfig.load(path)
    SLMConfig.load(path)

    assert len(_notices(caplog)) == 1
