# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The answer-quality script: judge mode against a stand-in daemon, and the
refusal to run an in-process engine on the account's own store."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from superlocalmemory.retrieval import judge_calibration_file as jcf

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "quality" / "answer_quality_eval.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("answer_quality_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gold(tmp_path: Path, answerable: int, unanswerable: int) -> Path:
    rows = [{"qid": f"A{i}", "question": f"answerable {i}", "gold_memory_ids": [f"m{i}"]}
            for i in range(answerable)]
    rows += [{"qid": f"U{i}", "question": f"unanswerable {i}", "answerable": False}
             for i in range(unanswerable)]
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _stand_in(answer_conf: float, empty_conf: float, threshold: float = 0.6):
    """A daemon whose judge is Laya: right memory first for answerable questions."""
    def request(method, path, body=None, **_kw):
        question = path.split("q=", 1)[1].split("&", 1)[0]
        index = question.rsplit("+", 1)[-1]
        answerable = question.startswith("answerable")
        conf = answer_conf if answerable else empty_conf
        results = [{"memory_id": f"m{index}" if answerable else "other", "fact_id": "f"}]
        return {"results": results, "answer_check_status": "judged",
                "calibration_id": "laya:org/model@abc:sufficiency-v1:top3",
                "reranker_status": "applied", "answer_confidence": conf,
                "abstained": conf < threshold}
    return request


class TestJudgeMode:
    def test_counts_each_outcome_and_writes_a_usable_file(
            self, tool, tmp_path, monkeypatch, capsys) -> None:
        import superlocalmemory.cli.daemon as daemon

        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

        # Laya is too cautious here: right answers at 0.5 are abstained at 0.6.
        monkeypatch.setattr(daemon, "daemon_request", _stand_in(0.5, 0.1))
        out = tmp_path / "cal.json"
        code = tool.main(["judges", "--gold", str(_gold(tmp_path, 25, 12)),
                          "--data-dir", str(tmp_path), "--write-calibration", str(out)])
        assert code == 0
        report = json.loads(capsys.readouterr().out)
        observed = report["providers"]["laya"]["observed"]
        assert observed["false_abstain"] == 25 and observed["correct_abstain"] == 12
        suggested = report["providers"]["laya"]["suggested"]
        assert suggested["correct_accept"] == 25 and suggested["false_accept"] == 0
        entries = jcf.parse_entries(json.loads(out.read_text(encoding="utf-8")))
        assert 0.1 < entries[("laya", "sufficiency-v1")] <= 0.5

    def test_too_few_questions_write_no_file(self, tool, tmp_path, monkeypatch, capsys) -> None:
        import superlocalmemory.cli.daemon as daemon

        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

        monkeypatch.setattr(daemon, "daemon_request", _stand_in(0.5, 0.1))
        out = tmp_path / "cal.json"
        tool.main(["judges", "--gold", str(_gold(tmp_path, 5, 2)),
                   "--data-dir", str(tmp_path), "--write-calibration", str(out)])
        report = json.loads(capsys.readouterr().out)
        assert report["calibration_file"]["written"] is False
        assert not out.exists()

    def test_a_silent_daemon_is_reported(self, tool, tmp_path, monkeypatch, capsys) -> None:
        import superlocalmemory.cli.daemon as daemon

        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

        monkeypatch.setattr(daemon, "daemon_request", lambda *a, **k: None)
        code = tool.main(["judges", "--gold", str(_gold(tmp_path, 1, 0)),
                          "--data-dir", str(tmp_path)])
        assert code == 4


class TestRetrievalModeRefusesTheOwnStore:
    def test_own_store_is_refused(self, tool, tmp_path, monkeypatch) -> None:
        own = tmp_path / "own"
        own.mkdir()
        (own / "memory.db").write_bytes(b"")
        monkeypatch.setattr(tool, "_owner_roots", lambda: {own.resolve()})
        code = tool.main(["retrieval", "--gold", str(_gold(tmp_path, 1, 0)),
                          "--data-dir", str(own)])
        assert code == 3

    def test_a_folder_without_a_store_is_refused(self, tool, tmp_path) -> None:
        code = tool.main(["retrieval", "--gold", str(_gold(tmp_path, 1, 0)),
                          "--data-dir", str(tmp_path / "nothing")])
        assert code == 2


class TestItRunsOnWindows:
    """Windows has no ``pwd``: the script failed to import there at all."""

    def test_it_imports_without_pwd(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "pwd", None)  # import pwd -> ImportError
        spec = importlib.util.spec_from_file_location("answer_quality_eval_nopwd", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module._account_home)

    def test_the_windows_home_comes_from_the_account_not_the_environment(
            self, tool, monkeypatch, tmp_path):
        import ctypes
        from types import SimpleNamespace

        profile = str(tmp_path / "profile")

        def folder_path(_hwnd, csidl, _token, _flags, buffer):
            assert csidl == 0x0028  # CSIDL_PROFILE
            buffer.value = profile
            return 0

        monkeypatch.setattr(tool.os, "name", "nt")
        monkeypatch.setattr(ctypes, "windll",
                            SimpleNamespace(shell32=SimpleNamespace(SHGetFolderPathW=folder_path)),
                            raising=False)
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "elsewhere"))
        assert tool._account_home() == Path(profile)


class TestMeasuresOnlyACompleteSearch:
    """A gate run measured the first questions while the embedding model was
    still warming: the same question ranked 7 three times, then dropped out
    once meaning-based search came online. The tool now waits for a complete
    search and counts any incomplete recall in its report."""

    class _Engine:
        def __init__(self, warming_for: int) -> None:
            self.calls = 0
            self.warming_for = warming_for

        def recall(self, *_a, **_k):
            from types import SimpleNamespace

            self.calls += 1
            missing = ("semantic",) if self.calls <= self.warming_for else ()
            return SimpleNamespace(incomplete_channels=missing)

    def test_it_waits_until_no_channel_is_warming(self, tool, monkeypatch) -> None:
        monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
        engine = self._Engine(warming_for=3)
        tool._wait_until_complete(engine, fast=None, timeout_s=30)
        assert engine.calls == 4

    def test_a_search_that_never_completes_is_refused(self, tool, monkeypatch) -> None:
        monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
        with pytest.raises(tool.EvalError, match="semantic"):
            tool._wait_until_complete(self._Engine(warming_for=10**9), fast=None, timeout_s=0.05)
