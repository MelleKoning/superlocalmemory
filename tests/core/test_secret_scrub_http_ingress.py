"""A memory is stored exactly as written, credentials included.

Keeping credentials is what SLM is for (the owner's rule, 2026-10-03): saving
never strips them. They are stripped only where text leaves this machine (see
tests/test_core/test_credentials_never_leave_the_machine.py).

The stored text must also equal the text the save was asked for, because the
background enrichment step re-checks the stored memory against the original
request; a stripped copy never matched, so those memories never finished.
"""

from __future__ import annotations

_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def test_write_queryable_keeps_the_memory_exactly_as_written(tmp_path) -> None:
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.ingestion_command import IngestionRequest
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    db.execute(
        "INSERT INTO profiles(profile_id, name, description) VALUES (?, ?, ?)",
        ("p1", "p1", "test profile"),
    )

    writer = build_immediate_admission_handler(db, profile_id="p1")
    content = f"Please store my AWS access key {_AWS_KEY} for the deploy pipeline."
    request = IngestionRequest(
        content=content,
        profile_id="p1",
        source_type="http-remember",
        idempotency_key="op-keep",
        trusted_actor_id="local-capability:test",
    )

    fact_ids = writer(request, "op-keep")
    assert fact_ids, "write_queryable should persist a queryable fact"

    fact_rows = db.execute(
        "SELECT content FROM atomic_facts WHERE fact_id = ?", (fact_ids[0],)
    )
    memory_rows = db.execute(
        "SELECT content FROM memories WHERE profile_id = ?", ("p1",)
    )
    assert fact_rows and memory_rows
    assert _AWS_KEY in str(fact_rows[0]["content"])
    # Byte-for-byte: the enrichment step compares exactly this.
    assert str(memory_rows[0]["content"]) == content
    assert "[REDACTED" not in str(memory_rows[0]["content"])
