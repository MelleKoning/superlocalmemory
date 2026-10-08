# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A save acknowledged just before the service is killed is enriched as soon as
the service is back, not 15 minutes later.

Found by the release check: kill -9 of the service while saves were arriving
left one save leased by the dead process. The restarted service could not
touch it until the 900 s lease ran out, so that memory stayed un-enriched for
up to 15 minutes. A lease whose owner process is provably gone is now released
at once; a live, unknown or foreign owner keeps its lease.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from superlocalmemory.core import ingestion_lease_owner as owners
from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M031_dead_letter_operations,
)


@pytest.fixture(autouse=True)
def _fresh_host_tag():
    clear = getattr(owners.host_tag, "cache_clear", lambda: None)
    clear()
    yield
    clear()


@pytest.fixture
def repository(tmp_path) -> IngestionOperationRepository:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M031_dead_letter_operations.apply(conn)
    return IngestionOperationRepository(db)


def _dead_owner() -> str:
    """The owner token a process that has since exited would have written."""
    code = ("from superlocalmemory.core.ingestion_lease_owner import owner_token;"
            "print(owner_token())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                         check=True)
    return out.stdout.strip()


def _leased(repository, key: str, owner: str) -> str:
    op = repository.create(IngestionRequest(
        content=f"Queryable evidence for {key}.", profile_id="default",
        source_type="http", idempotency_key=key))
    repository.db.execute(
        "UPDATE ingestion_operations SET state='enriching', lease_owner=?, "
        "lease_expires_at=?, attempt_count=1, "
        "queryable_fact_ids_json='[\"fact-visible\"]' WHERE operation_id=?",
        (owner, time.time() + 900, op.operation_id))
    return op.operation_id


def _due(repository) -> set[str]:
    return {op.operation_id for op in repository.list_materializable(limit=50)}


class TestLeasesOfADeadProcess:
    def test_a_dead_owner_s_lease_is_released_by_the_recovery_sweep(self, repository) -> None:
        orphan = _leased(repository, "crash:orphan", _dead_owner())
        assert orphan not in _due(repository)

        repository.reap_stuck_enriching()

        assert orphan in _due(repository), "a save of a killed service waited for its lease"
        claimed = repository.claim_enriching(orphan, owner=owners.owner_token(), lease_seconds=900)
        assert claimed.state.value == "enriching"

    def test_a_live_owner_keeps_its_lease(self, repository) -> None:
        mine = _leased(repository, "crash:live", owners.owner_token())
        repository.reap_stuck_enriching()
        assert mine not in _due(repository)

    def test_an_owner_it_cannot_identify_keeps_its_lease(self, repository) -> None:
        legacy = _leased(repository, "crash:legacy", "ingestion-worker:0123abcd")
        foreign = _leased(repository, "crash:foreign",
                          _dead_owner().replace(owners.host_tag(), "other-host", 1))
        repository.reap_stuck_enriching()
        due = _due(repository)
        assert legacy not in due and foreign not in due

    def test_a_reused_process_id_is_not_mistaken_for_the_owner(self, repository) -> None:
        token = owners.owner_token()
        pid_part, rest = token.split("@", 1)
        started, tail = rest.split(":", 1)
        reused = f"{pid_part}@{float(started) - 3600:.3f}:{tail}"   # same pid, older process
        op = _leased(repository, "crash:reused", reused)
        repository.reap_stuck_enriching()
        assert op in _due(repository)


def test_every_worker_writes_an_identifiable_owner(repository) -> None:
    command = IngestionCommand(repository, write_queryable=lambda *a, **k: None,
                               materialize=lambda *a, **k: None)
    assert owners.owner_is_dead(command._owner) is False
    assert owners.parse_owner(command._owner) is not None


def test_a_same_named_host_in_another_pid_namespace_never_steals_a_live_lease(monkeypatch) -> None:
    """Muse C1: two containers with the same host name and separate process
    namespaces share one store. A pid in the other namespace says nothing about
    the owner, so its lease is never released."""
    monkeypatch.setattr(owners, "_namespace_id", lambda: "boot-1/pidns-111", raising=False)
    owners.host_tag.cache_clear() if hasattr(owners.host_tag, "cache_clear") else None
    token = owners.owner_token()                  # written in container A
    monkeypatch.setattr(owners, "_namespace_id", lambda: "boot-1/pidns-222", raising=False)
    owners.host_tag.cache_clear() if hasattr(owners.host_tag, "cache_clear") else None
    monkeypatch.setattr("superlocalmemory.core.platform_utils.is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(owners, "_start_time", lambda _pid: 1.0)   # an unrelated process
    assert owners.owner_is_dead(token) is False, "a live lease in another namespace was released"


def test_an_unknown_machine_identity_never_releases(monkeypatch) -> None:
    monkeypatch.setattr(owners, "_namespace_id", lambda: None, raising=False)
    owners.host_tag.cache_clear() if hasattr(owners.host_tag, "cache_clear") else None
    token = owners.owner_token()
    monkeypatch.setattr("superlocalmemory.core.platform_utils.is_pid_alive", lambda _pid: False)
    assert owners.owner_is_dead(token) is False
