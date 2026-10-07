# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The second-edit refusal and "user action wins" agree on which cases count."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.open_correction import refuse_second_open_case
from superlocalmemory.core.remember_runtime import CanonicalMutationConflict


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE correction_cases (case_id TEXT, profile_id TEXT, "
              "predecessor_fact_id TEXT, successor_fact_id TEXT, status TEXT, "
              "reason_code TEXT, proposed_by_actor_kind TEXT, created_at TEXT)")
    return c


def _case(conn, status, reason, actor, case_id="c1") -> None:
    conn.execute("INSERT INTO correction_cases VALUES (?, 'default', 'f1', 's1', ?, ?, ?, '1')",
                 (case_id, status, reason, actor))


def test_a_pending_machine_proposal_does_not_block_the_users_edit(conn) -> None:
    _case(conn, "proposed", "temporal_contradiction", "host_attested")
    refuse_second_open_case(conn, "default", "f1")  # no conflict: the edit overtakes it


@pytest.mark.parametrize("status,reason,actor", [
    ("proposed", "direct_content_correction", "host_authenticated"),  # a person's edit
    ("applied", "consolidation_update", "host_attested"),             # already applied
])
def test_a_persons_case_or_an_applied_one_still_refuses(conn, status, reason, actor) -> None:
    _case(conn, status, reason, actor)
    with pytest.raises(CanonicalMutationConflict):
        refuse_second_open_case(conn, "default", "f1")


def test_a_person_case_behind_a_machine_one_still_refuses(conn) -> None:
    _case(conn, "proposed", "consolidation_supersede", "host_attested", "c0")
    _case(conn, "applied", "direct_content_correction", "host_authenticated", "c1")
    with pytest.raises(CanonicalMutationConflict, match="c1"):
        refuse_second_open_case(conn, "default", "f1")
