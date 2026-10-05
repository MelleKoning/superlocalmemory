# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saved views under GDPR: a view holds a person's own words (its query), so it
is exported with them and erased with them — and only theirs (issue #113)."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.views import ViewStore


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('alice', 'Alice')")
    mgr.store_memory(MemoryRecord(memory_id="m1", profile_id="alice", content="Alice info"))
    mgr.store_fact(AtomicFact(fact_id="f1", memory_id="m1", profile_id="alice",
                              content="Alice fact"))
    result = mr.apply_all(tmp_path / "learning.db", tmp_path / "memory.db")
    assert "M054_saved_views" in result["applied"]
    views = ViewStore(tmp_path / "learning.db")
    views.create("alice", name="Health", query="my blood test results")
    views.create("alice", name="Work", query="what shipped")
    views.create("bob", name="Bob", query="bob's things")
    return tmp_path


def _gdpr(root: Path) -> GDPRCompliance:
    return GDPRCompliance(DatabaseManager(root / "memory.db"), data_root=root)


def test_erasure_removes_the_persons_views_and_no_one_elses(root) -> None:
    counts = _gdpr(root).forget_profile("alice")
    assert counts["saved_views"] == 2
    views = ViewStore(root / "learning.db")
    assert views.list("alice") == ()
    assert [v.name for v in views.list("bob")] == ["Bob"]


def test_export_includes_every_view_with_its_query(root) -> None:
    exported = _gdpr(root).export_profile_data("alice")
    rows = exported["learning_signals"]["saved_views"]
    assert sorted(r["query"] for r in rows) == ["my blood test results", "what shipped"]
    assert {r["profile_id"] for r in rows} == {"alice"}


def test_a_deleted_workspace_takes_its_views_with_it(root, monkeypatch) -> None:
    """A workspace made later under the same name must not inherit them."""
    from superlocalmemory.server.routes import helpers

    monkeypatch.setattr(helpers, "DB_PATH", root / "memory.db")
    helpers.delete_profile_from_db("alice")
    views = ViewStore(root / "learning.db")
    assert views.list("alice") == () and len(views.list("bob")) == 1


def test_resetting_learning_data_keeps_views(root) -> None:
    """Views are the person's settings, not something the system learned."""
    from superlocalmemory.learning.database import LearningDatabase

    LearningDatabase(root / "learning.db").reset("alice")
    LearningDatabase(root / "learning.db").reset(None)
    assert len(ViewStore(root / "learning.db").list("alice")) == 2
