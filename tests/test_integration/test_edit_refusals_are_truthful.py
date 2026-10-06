# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Editing a memory that already has a correction open is refused, truthfully.

4.1.21 answered the second edit of a memory with HTTP 503 "canonical mutation
writer is temporarily unavailable" (the one-open-correction rule fired inside
the writer and was reported as an outage), and the tool interface turned that
into ``retryable: true``. On a 22k-fact store copy that was every second edit:
40 of 70 in one run. It is a refusal that never changes, so it is now a 409
that names the open case, and the tool says it is not retryable.

Composed path only: a REAL daemon and a REAL ``slm mcp`` child.
"""
from __future__ import annotations

import uuid

import pytest

from tests.test_integration.test_mcp_declared_kind_transport import _Lane

RUN = uuid.uuid4().hex[:6].upper()


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    lane = _Lane(tmp_path_factory.mktemp("edit-refusals"))
    try:
        lane.open_mcp()
        yield lane
    finally:
        lane.close_mcp()
        lane.daemon.stop(lane.foreign)


def _save(lane, tag: str) -> str:
    out = lane.tool("remember", content=f"Fixture record ED{tag}{RUN}: synthetic code E{tag}.",
                    idempotency_key=f"fixture-edit-{tag}-{RUN}")
    assert out["success"] is True, out
    return out["fact_ids"][0]


def test_a_second_edit_over_http_is_a_conflict_naming_the_open_case(lane):
    fid = _save(lane, "H")
    code, first = lane.daemon.request("PATCH", f"/api/memories/{fid}",
                                      {"content": f"Fixture record EDH{RUN}: code E1."})
    assert code == 202, first
    case = first["correction_case"]["case_id"]

    code, second = lane.daemon.request("PATCH", f"/api/memories/{fid}",
                                       {"content": f"Fixture record EDH{RUN}: code E2."})
    assert code == 409, second
    assert case in second["detail"] and "waiting for review" in second["detail"], second


def test_a_second_edit_through_the_tool_is_not_retryable(lane):
    fid = _save(lane, "M")
    first = lane.tool("update_memory", fact_id=fid, content=f"Fixture record EDM{RUN}: E3.")
    assert first.get("success") is True, first

    second = lane.tool("update_memory", fact_id=fid, content=f"Fixture record EDM{RUN}: E4.")
    assert second["success"] is False, second
    assert second["retryable"] is False and second["code"] == "CONFLICT", second
    assert first["correction_case"]["case_id"] in second["error"], second


def test_once_applied_the_old_version_points_to_the_current_one(lane):
    fid = _save(lane, "A")
    first = lane.tool("update_memory", fact_id=fid, content=f"Fixture record EDA{RUN}: E5.")
    case = first["correction_case"]
    applied = lane.tool("review_correction", case_id=case["case_id"], action="apply",
                        expected_version=case["version"])
    assert applied.get("success") is True, applied

    again = lane.tool("update_memory", fact_id=fid, content=f"Fixture record EDA{RUN}: E6.")
    assert again["success"] is False and again["retryable"] is False, again
    assert first["successor_fact_id"] in again["error"], again
    # Editing the current version works.
    current = lane.tool("update_memory", fact_id=first["successor_fact_id"],
                        content=f"Fixture record EDA{RUN}: E7.")
    assert current.get("success") is True, current
