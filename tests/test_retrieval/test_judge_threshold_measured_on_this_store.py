# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A judge's threshold can be set from a measurement on this store.

Laya abstained on 15 of 20 of the owner's questions and Jev on 8 of 20 with
the built-in thresholds, measured elsewhere. A threshold measured on this
store must be able to replace the built-in one, per judge, and a file that
does not hold a real measurement must never move any threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.retrieval import judge_calibration_file as jcf
from superlocalmemory.retrieval.judge_recipe import (
    CALIBRATIONS,
    RECIPE_V1,
    calibration_for,
    calibration_or_unmeasured,
)


def _entry(**overrides) -> dict:
    return {"backend": "laya", "recipe_id": RECIPE_V1.recipe_id, "threshold": 0.35,
            "answered": 30, "unanswered": 12, **overrides}


def _write(root: Path, *entries: dict, schema: str = jcf.SCHEMA) -> Path:
    path = root / jcf.FILE_NAME
    path.write_text(json.dumps({"schema": schema, "entries": list(entries)}), encoding="utf-8")
    return path


@pytest.fixture()
def data_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return tmp_path


class TestAMeasuredThresholdReplacesTheBuiltIn:
    def test_without_a_file_the_built_in_stands(self, data_dir) -> None:
        assert calibration_for("laya") == CALIBRATIONS[("laya", RECIPE_V1.recipe_id)]

    def test_each_judge_gets_its_own_measured_threshold(self, data_dir) -> None:
        _write(data_dir, _entry(threshold=0.35),
               _entry(backend="jev", threshold=0.62))
        assert calibration_for("laya").threshold == 0.35
        assert calibration_for("jev").threshold == 0.62
        assert calibration_for("laya").status == jcf.STATUS

    def test_a_judge_not_in_the_file_keeps_its_built_in(self, data_dir) -> None:
        _write(data_dir, _entry(backend="jev", threshold=0.62))
        assert calibration_for("laya") == CALIBRATIONS[("laya", RECIPE_V1.recipe_id)]
        assert calibration_for("jev-listwise") == CALIBRATIONS[
            ("jev-listwise", RECIPE_V1.recipe_id)]

    def test_another_recipe_is_not_moved(self, data_dir) -> None:
        _write(data_dir, _entry(recipe_id="sufficiency-v9", threshold=0.1))
        assert calibration_for("laya").threshold == CALIBRATIONS[
            ("laya", RECIPE_V1.recipe_id)].threshold

    def test_an_edited_file_is_read_again(self, data_dir) -> None:
        path = _write(data_dir, _entry(threshold=0.35))
        assert calibration_for("laya").threshold == 0.35
        _write(data_dir, _entry(threshold=0.4))
        import os
        stat = path.stat()
        os.utime(path, (stat.st_atime, stat.st_mtime + 5))
        assert calibration_for("laya").threshold == 0.4

    def test_the_jev_judge_uses_it(self, data_dir) -> None:
        from superlocalmemory.retrieval.jev_judge import JevSufficiencyJudge

        _write(data_dir, _entry(backend="jev", threshold=0.62))

        class _Keys:
            def has_key(self, provider):
                return False

        judge = JevSufficiencyJudge(provider="typesafe", key_store=_Keys())
        assert judge._threshold == 0.62
        assert judge._calibration_status == jcf.STATUS


class TestAFileWithoutARealMeasurementMovesNothing:
    @pytest.mark.parametrize("bad", [
        {"threshold": 1.5}, {"threshold": -0.1}, {"threshold": True},
        {"threshold": float("nan")}, {"answered": 5}, {"unanswered": 2},
        {"answered": "30"}, {"backend": "gpt"}, {"recipe_id": ""},
    ])
    def test_a_bad_entry_is_refused_whole(self, data_dir, bad) -> None:
        _write(data_dir, _entry(backend="jev", threshold=0.62), _entry(**bad))
        built_in = CALIBRATIONS[("jev", RECIPE_V1.recipe_id)]
        assert calibration_or_unmeasured("jev") == built_in

    def test_a_wrong_schema_is_refused(self, data_dir) -> None:
        _write(data_dir, _entry(), schema="something-else")
        assert calibration_for("laya") == CALIBRATIONS[("laya", RECIPE_V1.recipe_id)]

    def test_a_duplicate_pair_is_refused(self, data_dir) -> None:
        _write(data_dir, _entry(threshold=0.2), _entry(threshold=0.3))
        assert calibration_for("laya") == CALIBRATIONS[("laya", RECIPE_V1.recipe_id)]

    def test_unreadable_json_is_refused_and_logged(self, data_dir, caplog) -> None:
        (data_dir / jcf.FILE_NAME).write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert calibration_for("laya") == CALIBRATIONS[("laya", RECIPE_V1.recipe_id)]
        assert "built-in thresholds stay in force" in caplog.text
