# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The write path withholds a derived fact that changes its source, reversibly.

Real store, real ingestion command and lineage checkpoint; the extractor is
replaced by fixed derived facts so each case is exact.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M019_derivation_lineage,
)

SOURCE = "The Kestrel checkpoint was recalled at rank 1 in 2004.6 ms."
DERIVED = {
    "f-verbatim": SOURCE,
    "f-faithful": "The Kestrel checkpoint recall took 2004.6 ms",
    "f-date": "The recall happened on June 1st, 2004",
}


def _db(path: Path) -> DatabaseManager:
    db = DatabaseManager(path)
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M019_derivation_lineage.apply(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
        if "quarantined" not in columns:
            conn.execute("ALTER TABLE atomic_facts ADD COLUMN quarantined INTEGER DEFAULT 0")
    return db


def _ingest(db: DatabaseManager, facts: dict[str, str], key: str = "g03-1") -> None:
    def write_queryable(request: IngestionRequest, _operation_id: str) -> list[str]:
        db.execute(
            "INSERT INTO memories (memory_id,profile_id,content,session_id,speaker,role) "
            "VALUES (?,?,?, '', '', 'user')",
            (f"m-{key}", request.profile_id, request.content),
        )
        for fact_id, content in facts.items():
            db.execute(
                "INSERT INTO atomic_facts (fact_id,memory_id,profile_id,content,fact_type) "
                "VALUES (?,?,?,?,'semantic')",
                (fact_id, f"m-{key}", request.profile_id, content),
            )
        return list(facts)

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=write_queryable,
        materialize=lambda operation: operation.queryable_fact_ids,
        derivation_version="g03-test",
    )
    receipt = command.submit(IngestionRequest(
        content=SOURCE, profile_id="default", source_type="test", idempotency_key=key,
    ))
    assert command.materialize(receipt.operation_id).state.value == "complete"


def _state(db: DatabaseManager) -> dict[str, tuple[int, str]]:
    rows = db.execute(
        "SELECT f.fact_id, COALESCE(f.quarantined,0) AS q, l.unresolved_reason AS r "
        "FROM atomic_facts f LEFT JOIN derivation_lineage l "
        "ON l.object_type='fact' AND l.object_id=f.fact_id"
    )
    return {dict(r)["fact_id"]: (int(dict(r)["q"]), dict(r)["r"] or "") for r in rows}


def test_an_unfaithful_derived_fact_is_withheld_with_its_reason(tmp_path: Path) -> None:
    db = _db(tmp_path / "memory.db")
    _ingest(db, DERIVED)
    state = _state(db)
    assert state["f-verbatim"] == (0, ""), "the user's own words are never withheld"
    assert state["f-faithful"] == (0, "no_exact_span_in_raw_source")
    assert state["f-date"] == (1, "source_fidelity:number_became_date")
    # nothing is deleted or rewritten
    rows = db.execute("SELECT content FROM atomic_facts WHERE fact_id='f-date'")
    assert dict(rows[0])["content"] == DERIVED["f-date"]
    # quarantine is the one gate every recall channel passes through
    visible = {f.fact_id for f in db.get_facts_by_ids(list(DERIVED), "default")}
    assert visible == {"f-verbatim", "f-faithful"}


def _cli(db_path: Path, monkeypatch, **kwargs) -> int:
    from superlocalmemory.cli import fidelity_cmd

    class _Config:
        pass

    config = _Config()
    config.db_path = db_path
    monkeypatch.setattr("superlocalmemory.core.config.SLMConfig.load", lambda *a, **k: config)
    args = Namespace(json=True, profile="", limit=50, release="", **kwargs)
    return fidelity_cmd.cmd_db_fidelity(args)


def test_review_lists_withheld_and_older_facts(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "memory.db"
    db = _db(db_path)
    _ingest(db, DERIVED)
    # an older fact written before the check existed: live, never withheld
    db.execute(
        "INSERT INTO atomic_facts (fact_id,memory_id,profile_id,content,fact_type) "
        "VALUES ('f-old','m-g03-1','default','Recall was on 2004-06-16','semantic')"
    )
    assert _cli(db_path, monkeypatch) == 0
    out = json.loads(capsys.readouterr().out)
    assert [f["fact_id"] for f in out["withheld"]["facts"]] == ["f-date"]
    assert out["to_review"]["facts"] == [
        {"fact_id": "f-old", "profile_id": "default", "reasons": ["number_became_date"]},
    ]
    assert _state(db)["f-old"][0] == 0, "listing an old fact never changes it"


def test_release_is_offline_only_and_is_remembered(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "memory.db"
    db = _db(db_path)
    _ingest(db, DERIVED)

    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: True)
    assert _cli(db_path, monkeypatch) == 0  # listing is fine while running
    capsys.readouterr()
    from superlocalmemory.cli import fidelity_cmd

    refused = fidelity_cmd._release(str(db_path), "f-date")
    assert refused[0] is False and "Stop it first" in refused[1]
    assert _state(db)["f-date"][0] == 1

    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: False)
    ok, _ = fidelity_cmd._release(str(db_path), "f-date")
    assert ok is True
    assert _state(db)["f-date"] == (0, "source_fidelity_released:number_became_date")

    # a later checkpoint of the same fact does not withhold it again
    from superlocalmemory.core.derivation_lineage import capture_operation_lineage

    op = dict(db.execute("SELECT operation_id FROM ingestion_operations")[0])["operation_id"]
    capture_operation_lineage(db, operation_id=op, profile_id="default", raw_content=SOURCE,
                              fact_ids=tuple(DERIVED), derivation_version="g03-test")
    assert _state(db)["f-date"] == (0, "source_fidelity_released:number_became_date")


def test_an_older_damaged_fact_can_be_withheld_and_restored(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "memory.db"
    db = _db(db_path)
    _ingest(db, DERIVED)
    db.execute(
        "INSERT INTO atomic_facts (fact_id,memory_id,profile_id,content,fact_type) "
        "VALUES ('f-old','m-g03-1','default','Recall was on 2004-06-16','semantic')"
    )
    from superlocalmemory.cli import fidelity_cmd

    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: True)
    assert fidelity_cmd._withhold(str(db_path), "f-old")[0] is False, "never while running"
    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: False)
    assert fidelity_cmd._withhold(str(db_path), "f-faithful")[0] is False, "only failing facts"
    ok, message = fidelity_cmd._withhold(str(db_path), "f-old")
    assert ok and "--release f-old" in message
    assert _state(db)["f-old"] == (1, "source_fidelity:number_became_date")
    rows = db.execute("SELECT content FROM atomic_facts WHERE fact_id='f-old'")
    assert dict(rows[0])["content"] == "Recall was on 2004-06-16", "never rewritten"
    assert fidelity_cmd._release(str(db_path), "f-old")[0] is True
    assert _state(db)["f-old"] == (0, "source_fidelity_released:number_became_date")


def test_release_of_an_unwithheld_fact_is_refused(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "memory.db"
    _ingest(_db(db_path), DERIVED)
    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: False)
    from superlocalmemory.cli import fidelity_cmd

    ok, message = fidelity_cmd._release(str(db_path), "f-faithful")
    assert ok is False and "nothing to release" in message


@pytest.mark.parametrize("fact", [
    "Publish with approval from the release owner",
    "Ship the Android build before the iOS build",
])
def test_polarity_damage_is_withheld(tmp_path: Path, fact: str) -> None:
    from superlocalmemory.core.source_fidelity_guard import withhold_if_unfaithful

    db = _db(tmp_path / "memory.db")
    source = ("Never publish without approval from the release owner. "
              "Ship the iOS build before the Android build.")
    db.execute("INSERT INTO memories (memory_id,profile_id,content) VALUES ('m','default',?)",
               (source,))
    db.execute("INSERT INTO atomic_facts (fact_id,memory_id,profile_id,content,fact_type) "
               "VALUES ('f','m','default',?,'semantic')", (fact,))
    reason = withhold_if_unfaithful(db, profile_id="default", fact_id="f", content=fact,
                                    raw_content=source)
    assert reason and reason.startswith("source_fidelity:")
    assert _state(db)["f"][0] == 1
