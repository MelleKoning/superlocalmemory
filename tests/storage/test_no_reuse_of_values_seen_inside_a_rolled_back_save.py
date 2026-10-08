# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A value computed inside an open transaction is never kept for reuse.

The entity lookup and the graph's store counts keep a value while the store's
commit signature (storage/store_signature.py) is unchanged. Inside an open
transaction a read sees rows that are not committed; if that transaction then
rolls back, the files are as they were, the signature matches, and the kept
value would name an entity that does not exist.
"""

from __future__ import annotations

import pytest

from superlocalmemory.encoding.entity_lookup_index import EntityLookupIndex
from superlocalmemory.encoding.entity_resolver import EntityResolver
from superlocalmemory.storage import schema, store_signature
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import CanonicalEntity

P = "default"


class _Rollback(Exception):
    pass


def _db(tmp_path) -> DatabaseManager:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    db.store_entity(CanonicalEntity(profile_id=P, canonical_name="Kept Entity",
                                    entity_type="person"))
    return db


def test_a_save_finds_its_own_new_entity_and_a_rollback_forgets_it(tmp_path) -> None:
    db = _db(tmp_path)
    resolver = EntityResolver(db)
    assert set(resolver.lookup(["Kept Entity", "Phantom Entity"], P)) == {"Kept Entity"}
    with pytest.raises(_Rollback):
        with db.transaction():
            db.store_entity(CanonicalEntity(profile_id=P, canonical_name="Phantom Entity",
                                            entity_type="person"))
            assert "Phantom Entity" in resolver.lookup(["Phantom Entity"], P)
            raise _Rollback
    assert set(resolver.lookup(["Kept Entity", "Phantom Entity"], P)) == {"Kept Entity"}


def test_the_kept_rows_are_not_used_inside_a_transaction(tmp_path) -> None:
    db = _db(tmp_path)
    index = EntityLookupIndex(db)
    assert index.get(P) is not None
    with db.transaction():
        assert index.get(P) is None  # callers read the store itself


def test_no_signature_is_given_inside_a_transaction(tmp_path) -> None:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    assert store_signature.of_db(db) is not None
    with db.transaction():
        assert store_signature.of_db(db) is None
    assert store_signature.of_db(db) is not None
