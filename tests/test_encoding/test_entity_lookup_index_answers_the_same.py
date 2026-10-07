# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Recall's entity lookup answers exactly as the table scans did, without them.

``EntityResolver.lookup`` scanned the entity and alias tables for every name in
every question (46 ms a recall on a 22k-fact store). It now keeps the rows in
memory while nothing has been committed. These tests pin that every answer is
the one the original three queries give, including the cases where a scan's
row order or SQLite's ASCII-only LOWER() decide it, and that any commit is seen.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.encoding import entity_resolver as er
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import CanonicalEntity, EntityAlias

P = "default"
NAMES = ["Quorvantel Brask", "quorvantel brask", "Élodie Marchand", "ÉLODIE MARCHAND",
         "Hollowmere", "Kestrel Harbour", "Brindlemoor", "Acme Billing", "Nadir Okonkwo",
         "Ångström Labs", "Torvane Optics"]
ALIASES = [("Hollowmere", "Hollow"), ("Kestrel Harbour", "Kestrel"), ("Brindlemoor", "Kestrel"),
           ("Acme Billing", "acme"), ("Nadir Okonkwo", "Nadir"),
           # SQLite lowers ASCII only: this alias, not the name above, is what
           # an exact lookup of "ÅNGSTRÖM LABS" finds.
           ("Torvane Optics", "ÅNGSTRÖM LABS")]
QUESTIONS = ["Quorvantel Brask", "QUORVANTEL BRASK", "Élodie Marchand", "élodie marchand",
             "Hollow", "HOLLOW", "Kestrel", "Acme", "Hollowmer", "Kestrel Harbor", "Brindlemore",
             "Nadir Okonkw", "Unknown Person", "Zzyzx Qwv", "acme billing", "ÅNGSTRÖM LABS"]


@pytest.fixture()
def resolver(tmp_path):
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    ids = {}
    for name in NAMES:
        ids[name] = db.store_entity(CanonicalEntity(profile_id=P, canonical_name=name,
                                                    entity_type="person"))
    for owner, alias in ALIASES:
        db.store_alias(EntityAlias(entity_id=ids[owner], alias=alias), P)
    return er.EntityResolver(db, llm=None) if "llm" in er.EntityResolver.__init__.__code__.co_varnames \
        else er.EntityResolver(db)


def _original(res, name: str) -> str | None:
    entity = res._db.get_entity_by_name(name, P)
    if entity is not None:
        return entity.entity_id
    alias = res._alias_lookup(name, P)
    if alias is not None:
        return alias
    match_id, score = res._fuzzy_match(name, P)
    return match_id if match_id is not None and score >= er.JARO_WINKLER_AUTO_MERGE else None


def test_every_answer_is_the_original_one(resolver) -> None:
    got = resolver.lookup(QUESTIONS, P)
    for q in QUESTIONS:
        if er._usable_mention(q) is None:
            continue
        assert got.get(q) == _original(resolver, q), q
    assert any(got.get(q) for q in QUESTIONS)  # not vacuous: names were found


def test_a_name_added_by_another_connection_is_found_next_time(resolver) -> None:
    assert "Saffronwick" not in resolver.lookup(["Saffronwick"], P)
    conn = sqlite3.connect(resolver._db.db_path)
    conn.execute("INSERT INTO canonical_entities (entity_id, profile_id, canonical_name, "
                 "entity_type, first_seen, last_seen, fact_count) VALUES "
                 "('e-saffron', ?, 'Saffronwick', 'person', 'x', 'x', 0)", (P,))
    conn.commit()
    conn.close()
    assert resolver.lookup(["Saffronwick"], P) == {"Saffronwick": "e-saffron"}


def test_the_tables_are_not_scanned_again_while_nothing_changed(resolver, monkeypatch) -> None:
    resolver.lookup(QUESTIONS, P)
    calls = []
    real = resolver._db.execute

    def spy(sql, *a, **k):
        calls.append(sql)
        return real(sql, *a, **k)

    monkeypatch.setattr(resolver._db, "execute", spy)
    resolver.lookup(["Hollow", "Hollowmer", "Unknown Person"], P)
    assert calls == []
