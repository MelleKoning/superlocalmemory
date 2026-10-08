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


class TestAnswerNotStoredIsNotAMiss:
    """A labelled answer missing from the measured store is reported as
    "answer not stored" with the labelled ids' write dates, not as a miss; and
    the per-question rows keep every returned id up to --limit (identity gates
    compare the top 10, not the top 5)."""

    class _Engine:
        profile_id = "default"
        _config = None

        def __init__(self, ids: list[str]) -> None:
            self.ids = ids

        def recall(self, _query, limit=10, **_k):
            from types import SimpleNamespace

            results = [SimpleNamespace(fact=SimpleNamespace(memory_id=m, fact_id=f"f-{m}"))
                       for m in self.ids[:limit]]
            return SimpleNamespace(results=results, query_type="t", incomplete_channels=(),
                                   reranker_status="applied", no_confident_match=False)

        def close(self) -> None:
            pass

    def test_presence_is_checked_and_rows_keep_the_top_ten(
            self, tool, tmp_path, monkeypatch, capsys) -> None:
        import sqlite3

        import superlocalmemory.core.recall_pipeline as rp

        store = tmp_path / "copy"
        store.mkdir()
        conn = sqlite3.connect(store / "memory.db")
        conn.executescript("""
            CREATE TABLE memories (memory_id TEXT PRIMARY KEY, profile_id TEXT, created_at TEXT);
            CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT, created_at TEXT);
            INSERT INTO memories VALUES ('m0', 'default', '2026-08-26 10:00:00');
            INSERT INTO memories VALUES ('m1', 'default', '2026-09-01 10:00:00');
        """)
        conn.commit()
        conn.close()
        returned = [f"x{i}" for i in range(11)] + ["m1"]       # m1 lands at rank 12
        returned[7] = "m0"                                      # m0 at rank 8
        monkeypatch.setattr(tool, "_owner_roots", lambda: set())
        monkeypatch.setattr(tool, "_build_engine", lambda *_a: self._Engine(returned))
        monkeypatch.setattr(tool, "_wait_until_complete", lambda *_a: None)
        monkeypatch.setattr(rp, "resolve_hot_path_fast", lambda *_a: None)
        out = tmp_path / "rows.jsonl"
        gold = _gold(tmp_path, 3, 1)                           # A0 m0, A1 m1, A2 m2 (absent)
        code = tool.main(["retrieval", "--gold", str(gold), "--data-dir", str(store),
                          "--limit", "10", "--out", str(out)])
        assert code == 0
        report = json.loads(capsys.readouterr().out)
        assert "answer_presence" in report, "the store was not checked for the answers"
        presence = report["answer_presence"]
        assert presence["checked"] == 3 and presence["stored"] == 2
        assert presence["answer_not_stored"] == ["A2"]
        assert presence["retrieval_misses"] == ["A1"]
        assert presence["misses_explained_by_storage"] == ["A2"]
        assert report["stored_only"]["n"] == 2
        rows = {r["qid"]: r for r in map(json.loads, out.read_text().splitlines())}
        assert rows["A0"]["answer_presence"]["earliest_created_at"] == "2026-08-26 10:00:00"
        assert rows["A2"]["answer_presence"]["status"] == "not_stored"
        assert "answer_presence" not in rows["U0"]
        assert len(rows["A0"]["top"]) == 10
