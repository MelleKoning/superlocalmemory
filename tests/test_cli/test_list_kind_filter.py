# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm list --kind`` (LLD/WP8 4.1.19) — same ``list_recent_facts`` helper
the MCP tool uses, so the two surfaces cannot disagree.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import commands
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


class _FakeEngine:
    def __init__(self, db) -> None:
        self._db = db
        self.profile_id = "default"

    def initialize(self) -> None:
        pass


def _save(db, content, *, kind=None, source=None) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                      fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source=source)
    return db.store_fact(fact)


@pytest.fixture()
def fake_engine(tmp_path, monkeypatch):
    from superlocalmemory.core import engine as engine_mod

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    engine = _FakeEngine(db)
    monkeypatch.setattr(engine_mod, "MemoryEngine", lambda *a, **k: engine)
    return engine


def test_list_kind_filter_keeps_only_matches(fake_engine, capsys) -> None:
    _save(fake_engine._db, "rule one", kind="rule", source="user")
    _save(fake_engine._db, "a decision", kind="decision", source="user")
    commands.cmd_list(Namespace(json=True, limit=10, kind="rule"))
    out = json.loads(capsys.readouterr().out)
    contents = [r["content"] for r in out["data"]["results"]]
    assert contents == ["rule one"]


def test_list_defaults_to_no_kind_filter_when_attribute_absent(fake_engine, capsys) -> None:
    # The existing AST test constructs Namespace(json=True, limit=1) with no
    # `kind` attribute at all — cmd_list must not break on that shape.
    _save(fake_engine._db, "anything")
    commands.cmd_list(Namespace(json=True, limit=1))
    out = json.loads(capsys.readouterr().out)
    assert out["data"]["count"] == 1


def test_list_refuses_an_unknown_kind_before_any_read(fake_engine, capsys) -> None:
    _save(fake_engine._db, "anything")
    with pytest.raises(SystemExit) as exited:
        commands.cmd_list(Namespace(json=True, limit=10, kind="not-a-real-kind"))
    assert exited.value.code == 2


def test_list_text_shows_id_and_kind_label(fake_engine, capsys) -> None:
    """Audit 4.1.20 L1: the help promises IDs; the text view printed none, and
    showed the internal fact type ("semantic") instead of the memory kind."""
    fact_id = _save(fake_engine._db, "We decided to ship on Fridays",
                    kind="decision", source="user")

    commands.cmd_list(Namespace(json=False, limit=10, kind=""))
    out = capsys.readouterr().out

    assert f"id: {fact_id}" in out
    assert "Decision: We decided to ship on Fridays" in out
    assert "(semantic)" not in out
    assert "slm delete <id>" in out


def test_list_help_matches_what_text_mode_shows(tmp_path) -> None:
    import os
    import subprocess
    import sys

    env = dict(os.environ, HOME=str(tmp_path), SLM_DATA_DIR=str(tmp_path / "slm"))
    out = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "list", "--help"],
        capture_output=True, text=True, timeout=60, env=env,
    ).stdout
    assert "ID" in out and "kind" in out
