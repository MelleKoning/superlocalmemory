"""The store check after an upgrade only reads; a repair the owner starts backs up
first and changes nothing when the backup cannot be made."""
import sqlite3
import time

import pytest

from superlocalmemory.storage import store_check


def _wait(root, *statuses, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = store_check.read_state(root)
        if not state["running"] and state.get("status") in statuses:
            return state
        time.sleep(0.05)
    raise AssertionError(f"still {store_check.read_state(root)}")


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "memory.db"
    sqlite3.connect(db).execute("CREATE TABLE t(x)").connection.commit()
    return db


def test_findings_count_only_what_a_repair_acts_on():
    plan = {
        "vector_parity": {"stale_vectors": 11, "lance": {"orphans": 7}},
        "unreachable_vectors": 2,
        "orphans": [{"rows": 3, "action": "remove"}, {"rows": 9, "action": "keep"}],
        "erased_text": {"erased_facts": 40, "event_previews": 4, "journal_text": 1},
        "failed_obligations": {"proven_erased": 1, "retryable": 2, "needs_review": 5},
        "memories_without_own_fact": {"to_repair": 1, "held_x": 3},
    }
    assert store_check.findings(plan) == {
        "stale_vectors": 11, "lance_orphans": 7, "unreachable_vectors": 2,
        "leftover_rows": 3, "erased_words": 5, "unfinished_updates": 3,
        "memories_without_fact": 1,
    }
    # Not checked here (None) counts as nothing found, never as an error.
    assert sum(store_check.findings({"vector_parity": {"stale_vectors": None}}).values()) == 0
    assert set(store_check.FINDINGS) == set(store_check.findings({}))


def test_the_check_after_an_upgrade_runs_once_per_version_and_only_reads(store, monkeypatch):
    monkeypatch.setattr(store_check, "FIRST_CHECK_DELAY_S", 0.0)
    monkeypatch.setattr("superlocalmemory.storage.integrity_scan.plan",
                        lambda conn, **kw: {"vector_parity": {"stale_vectors": 4}})
    before = store.read_bytes()
    root = store.parent
    assert store_check.check_after_upgrade(store, root, "4.1.23")
    state = _wait(root, "checked")
    assert (state["to_repair"], state["checked_version"]) == (4, "4.1.23")
    assert store.read_bytes() == before
    assert not store_check.check_after_upgrade(store, root, "4.1.23")
    assert store_check.check_after_upgrade(store, root, "4.1.24")
    _wait(root, "checked")


def test_a_repair_backs_up_first_then_repairs_then_checks_again(store, monkeypatch):
    order = []

    class Backups:
        def __init__(self, **kw):
            pass

        def create_backup(self, label=None):
            order.append(("backup", label))
            return "memory-before-repair.db"

    class Repair:
        def __init__(self, db_path, engine=None):
            pass

        def apply(self):
            order.append("repair")
            return {"status": "finished", "run_id": "r1"}

    monkeypatch.setattr("superlocalmemory.infra.backup.BackupManager", Backups)
    monkeypatch.setattr("superlocalmemory.storage.integrity_repair.Repair", Repair)
    monkeypatch.setattr("superlocalmemory.storage.integrity_scan.plan", lambda conn, **kw: {})
    assert store_check.start_repair(store, store.parent, "4.1.23")
    state = _wait(store.parent, "repaired")
    assert order == [("backup", "before-repair"), "repair"]
    assert state["backup"] == "memory-before-repair.db" and state["to_repair"] == 0


def test_no_backup_means_nothing_is_changed(store, monkeypatch):
    class Backups:
        def __init__(self, **kw):
            pass

        def create_backup(self, label=None):
            return ""

    def never(*a, **kw):
        raise AssertionError("repair ran without a backup")

    monkeypatch.setattr("superlocalmemory.infra.backup.BackupManager", Backups)
    monkeypatch.setattr("superlocalmemory.storage.integrity_repair.Repair", never)
    assert store_check.start_repair(store, store.parent, "4.1.23")
    state = _wait(store.parent, "repair_failed")
    assert "nothing was changed" in state["error"]


def test_only_one_check_or_repair_runs_at_a_time(store, monkeypatch):
    gate = __import__("threading").Event()
    monkeypatch.setattr("superlocalmemory.storage.integrity_scan.plan",
                        lambda conn, **kw: gate.wait(5) and {})
    assert store_check.start_check(store, store.parent, "4.1.23")
    assert not store_check.start_repair(store, store.parent, "4.1.23")
    assert store_check.read_state(store.parent)["running"]
    gate.set()
    _wait(store.parent, "checked")
