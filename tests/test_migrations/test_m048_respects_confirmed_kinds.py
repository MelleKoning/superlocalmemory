# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The plan re-read must never demote a plan a person filed as a plan.

M048 runs every maintenance cycle and moves anything filed as a plan whose
wording does not read like one. Once a user (or the agent writing on their
behalf) has said "this is a plan", that is the authority: the wording rule is a
guess, the person is not. Rows whose kind is only a model's suggestion, or that
have no kind at all, are re-read exactly as before.
"""

from __future__ import annotations

import contextlib
import sqlite3

import pytest

from superlocalmemory.storage.migrations import (
    M048_upcoming_holds_only_what_is_upcoming as M048,
)

#: Reads as finished work, so the wording rule would demote it.
_NOT_PLAN_WORDING = "We shipped the fix yesterday"

_WITH_KINDS = (
    "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT, "
    "fact_type TEXT NOT NULL DEFAULT 'semantic' CHECK (fact_type IN "
    "('episodic','semantic','opinion','prospective')), "
    "memory_kind TEXT, memory_kind_source TEXT, memory_kind_confidence REAL, "
    "memory_kind_recipe TEXT, memory_kind_at TEXT)"
)
_WITHOUT_KINDS = (
    "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT, "
    "fact_type TEXT NOT NULL DEFAULT 'semantic' CHECK (fact_type IN "
    "('episodic','semantic','opinion','prospective')))"
)


def _store(tmp_path, ddl: str) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute(ddl)
    conn.commit()
    return conn


def _seed_kinds(conn, rows) -> None:
    conn.executemany(
        "INSERT INTO atomic_facts (fact_id, content, fact_type, memory_kind, "
        "memory_kind_source) VALUES (?, ?, 'prospective', ?, ?)", rows)
    conn.commit()


def _type(conn, fact_id: str) -> str:
    return conn.execute(
        "SELECT fact_type FROM atomic_facts WHERE fact_id=?", (fact_id,)
    ).fetchone()[0]


@pytest.mark.parametrize("source", ["user", "caller"])
def test_user_confirmed_plan_is_not_demoted(tmp_path, source) -> None:
    conn = _store(tmp_path, _WITH_KINDS)
    try:
        _seed_kinds(conn, [("f1", _NOT_PLAN_WORDING, "prospective", source)])

        M048.apply(conn)

        assert _type(conn, "f1") == "prospective"
        # And the standing guard does not report it as drift every cycle.
        assert M048.verify(conn) is True
    finally:
        conn.close()


def test_the_maintenance_path_also_leaves_a_confirmed_plan_alone(tmp_path) -> None:
    conn = _store(tmp_path, _WITH_KINDS)
    try:
        _seed_kinds(conn, [("f1", _NOT_PLAN_WORDING, "prospective", "user"),
                           ("f2", _NOT_PLAN_WORDING, "prospective", None)])

        @contextlib.contextmanager
        def opener():
            yield conn

        M048.apply(open_connection=opener)

        assert _type(conn, "f1") == "prospective"
        assert _type(conn, "f2") != "prospective"
    finally:
        conn.close()


@pytest.mark.parametrize("source", [None, "model:laya", "model:jev", "model:llm",
                                    "rules", "legacy", ""])
def test_unconfirmed_plan_without_planning_language_is_demoted(tmp_path, source) -> None:
    conn = _store(tmp_path, _WITH_KINDS)
    try:
        _seed_kinds(conn, [("f1", _NOT_PLAN_WORDING, "prospective", source)])
        assert M048.verify(conn) is False

        M048.apply(conn)

        assert _type(conn, "f1") != "prospective"
        assert M048.verify(conn) is True
    finally:
        conn.close()


def test_store_without_kind_columns_behaves_as_before(tmp_path) -> None:
    conn = _store(tmp_path, _WITHOUT_KINDS)
    try:
        conn.executemany(
            "INSERT INTO atomic_facts VALUES (?, ?, 'prospective')",
            [("f1", _NOT_PLAN_WORDING), ("f2", "Dentist appointment tomorrow at 10:30")])
        conn.commit()
        assert M048.verify(conn) is False

        M048.apply(conn)

        assert _type(conn, "f1") != "prospective"
        assert _type(conn, "f2") == "prospective"
        assert M048.verify(conn) is True
    finally:
        conn.close()


def test_the_ddl_string_is_unchanged() -> None:
    """Editing it would change the recorded hash and fail every upgraded store."""
    import hashlib

    assert hashlib.sha256(M048.DDL.encode("utf-8")).hexdigest() == (
        hashlib.sha256((
            "\n-- No schema change. apply() re-reads the content of every fact typed\n"
            "-- 'prospective' and demotes the ones whose wording does not describe something\n"
            "-- still ahead.\n"
        ).encode("utf-8")).hexdigest()
    )
