# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Turning the store-time consistency check back on must stick, and the one-time
switch-off of the old stored default must say so.

Every config saved by 4.1.0-4.1.17 holds the whole ``math`` section — eleven
keys, ``sheaf_at_encoding: true`` among them — because save() writes the full
section. That is a stored default, not a choice, and 4.1.18 reads it as off
once. A ``math`` section someone wrote by hand (the docs say: set
``sheaf_at_encoding`` to true under ``math``) is a choice, and must be kept.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from superlocalmemory.core import config as config_mod
from superlocalmemory.core.config import SLMConfig

#: Exactly what every 4.1.x save() wrote for ``math`` (checked against the
#: v4.1.0, v4.1.5, v4.1.10 and v4.1.17 tags).
OLD_DUMP = {
    "fisher_temperature": 15.0, "fisher_bayesian_update": True,
    "fisher_mode": "simplified", "langevin_dt": 0.005, "langevin_temperature": 0.3,
    "langevin_persist_positions": True, "langevin_weight_range": [0.0, 1.0],
    "ebbinghaus_langevin_coupling_enabled": False, "sheaf_at_encoding": True,
    "sheaf_contradiction_threshold": 0.45, "sheaf_max_edges_per_check": 200,
}


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    from superlocalmemory.core import config_upgrades

    monkeypatch.setattr(config_upgrades, "_sheaf_switch_off_reported", False)


def _write(tmp_path: Path, math: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "a", "math": math}), encoding="utf-8")
    return path


def test_a_hand_written_math_section_turning_it_on_is_respected(tmp_path):
    path = _write(tmp_path, {"sheaf_at_encoding": True})
    assert SLMConfig.load(path).math.sheaf_at_encoding is True


def test_a_hand_written_section_survives_a_save(tmp_path):
    path = _write(tmp_path, {"sheaf_at_encoding": True, "sheaf_contradiction_threshold": 0.5})
    SLMConfig.load(path).save(path)
    assert SLMConfig.load(path).math.sheaf_at_encoding is True


def test_the_old_stored_default_is_switched_off_with_a_warning(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=config_mod.logger.name)
    path = _write(tmp_path, OLD_DUMP)
    assert SLMConfig.load(path).math.sheaf_at_encoding is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    text = " ".join(r.getMessage() for r in warnings)
    assert "math.sheaf_at_encoding" in text
    assert "true" in text.lower(), "the warning says how to turn it back on"


def test_the_warning_is_given_once_not_on_every_load(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=config_mod.logger.name)
    path = _write(tmp_path, OLD_DUMP)
    for _ in range(3):
        SLMConfig.load(path)
    hits = [r for r in caplog.records if "math.sheaf_at_encoding" in r.getMessage()]
    assert len(hits) == 1


def test_an_old_dump_that_already_had_it_off_is_not_reported(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=config_mod.logger.name)
    path = _write(tmp_path, {**OLD_DUMP, "sheaf_at_encoding": False})
    assert SLMConfig.load(path).math.sheaf_at_encoding is False
    assert not [r for r in caplog.records if "math.sheaf_at_encoding" in r.getMessage()]


def test_the_docs_say_what_turning_it_off_changes_in_mode_a():
    doc = (Path(__file__).resolve().parents[2] / "docs" / "configuration.md").read_text(encoding="utf-8")
    section = doc[doc.index("## Consistency Checking at Store Time"):]
    section = section[: section.index("\n## ", 3)]
    assert "Mode A" in section
    assert "sheaf_at_encoding" in section
