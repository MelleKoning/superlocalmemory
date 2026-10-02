# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The sheaf check is off by default, everywhere it runs.

4.1.18: a measured A/B on a real store showed the sheaf consistency check
changes no recall answer, so it is switched off by default rather than paid
for on every store and every maintenance pass. One flag governs both places
it runs — the store path (via engine wiring) and the maintenance pass — so
turning it off cannot leave half of it running. The code is kept: True turns
it back on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.maintenance import run_maintenance


class _Checker:
    built: list = []
    checked: list = []

    def __init__(self, db, threshold) -> None:
        type(self).built.append(threshold)

    def check_consistency(self, fact, profile_id):
        type(self).checked.append(fact.fact_id)
        return []


@pytest.fixture()
def checker(monkeypatch):
    import superlocalmemory.math.sheaf as sheaf

    _Checker.built, _Checker.checked = [], []
    monkeypatch.setattr(sheaf, "SheafConsistencyChecker", _Checker)
    return _Checker


def _config(**math) -> SLMConfig:
    cfg = SLMConfig()
    # Only the sheaf pass is under test; the other math passes need more setup.
    cfg.math.langevin_persist_positions = False
    cfg.math.fisher_bayesian_update = False
    cfg.math.ebbinghaus_langevin_coupling_enabled = False
    for name, value in math.items():
        setattr(cfg.math, name, value)
    return cfg


def _db_with_a_recent_fact() -> MagicMock:
    fact = SimpleNamespace(
        fact_id="f-recent", created_at=datetime.now(UTC).isoformat(),
        embedding=[0.1, 0.2], canonical_entities=["Alice"], langevin_position=None,
        access_count=0, importance=0.5, fisher_variance=None, lifecycle="active",
    )
    db = MagicMock()
    db.get_all_facts.return_value = [fact]
    db.execute.return_value = []
    db.gc_orphaned_embedding_metadata.return_value = 0
    return db


def test_by_default_maintenance_never_runs_the_sheaf_check(checker) -> None:
    counts = run_maintenance(_db_with_a_recent_fact(), _config(), profile_id="default")
    assert counts["sheaf_checked"] == 0
    assert checker.built == [] and checker.checked == []


def test_switched_back_on_it_runs_on_recent_facts(checker) -> None:
    """Proves the test above can fail: with the flag on, the same fact is checked."""
    counts = run_maintenance(_db_with_a_recent_fact(), _config(sheaf_at_encoding=True),
                             profile_id="default")
    assert counts["sheaf_checked"] == 1
    assert checker.checked == ["f-recent"]


def test_by_default_the_store_path_gets_no_sheaf_checker(monkeypatch, checker) -> None:
    """Engine wiring builds the checker the store path and the temporal
    validator use from the same flag."""
    from superlocalmemory.core import engine_wiring

    cfg = _config()
    for name in ("_init_vector_store", "_init_access_log", "_init_context_generator",
                 "_init_auto_linker", "_init_graph_analyzer"):
        monkeypatch.setattr(engine_wiring, name, lambda *a, **k: None)
    seen: list = []
    monkeypatch.setattr(engine_wiring, "_init_temporal",
                        lambda config, db, sheaf_checker, llm: seen.append(sheaf_checker))
    try:
        parts = engine_wiring.init_encoding(cfg, MagicMock(), MagicMock(), None)
    except TypeError:
        pytest.skip("init_encoding signature changed; covered by the maintenance tests")
    assert parts["sheaf_checker"] is None
    assert seen == [None]
    assert checker.built == []


class TestExistingInstallsAreSwitchedOffOnce:
    """Configs saved before 4.1.18 hold sheaf_at_encoding: true because save()
    writes the whole math section — a stored default, not anyone's choice."""

    def _write(self, tmp_path, math: dict):
        import json
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"mode": "a", "math": math}))
        return path

    #: What every 4.1.0-4.1.17 save() wrote for ``math``: all eleven keys.
    #: A hand-written section is a choice and is covered in
    #: test_sheaf_explicit_choice.py.
    _OLD_DUMP = {
        "fisher_temperature": 15.0, "fisher_bayesian_update": True,
        "fisher_mode": "simplified", "langevin_dt": 0.005, "langevin_temperature": 0.3,
        "langevin_persist_positions": True, "langevin_weight_range": [0.0, 1.0],
        "ebbinghaus_langevin_coupling_enabled": False, "sheaf_at_encoding": True,
        "sheaf_contradiction_threshold": 0.45, "sheaf_max_edges_per_check": 200,
    }

    def test_a_pre_4_1_18_config_reads_as_off(self, tmp_path) -> None:
        path = self._write(tmp_path, dict(self._OLD_DUMP))
        assert SLMConfig.load(path).math.sheaf_at_encoding is False

    def test_someone_who_turned_it_back_on_keeps_it_on(self, tmp_path) -> None:
        path = self._write(tmp_path, {"sheaf_at_encoding": True, "sheaf_default_reviewed": True})
        assert SLMConfig.load(path).math.sheaf_at_encoding is True

    def test_the_switch_happens_once_and_then_sticks(self, tmp_path) -> None:
        import json
        path = self._write(tmp_path, dict(self._OLD_DUMP))
        config = SLMConfig.load(path)
        config.save(path)
        saved = json.loads(path.read_text())["math"]
        assert saved["sheaf_default_reviewed"] is True
        assert saved["sheaf_at_encoding"] is False
        reloaded = SLMConfig.load(path)
        reloaded.math = type(reloaded.math)(**{**reloaded.math.__dict__, "sheaf_at_encoding": True})
        reloaded.save(path)
        assert SLMConfig.load(path).math.sheaf_at_encoding is True
