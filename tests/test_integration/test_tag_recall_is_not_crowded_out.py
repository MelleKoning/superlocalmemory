# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A tag-filtered recall finds the memory behind many closer, untagged ones.

Mirrors ``test_kind_recall_is_not_crowded_out.py`` for ``tags`` (4.1.22 G05):
fusion takes each channel's top candidates, so when enough OTHER memories
match a question's words better, the one memory carrying the asked-for tag
never became a candidate at all, and a filter applied only after fusion comes
back empty although the memory is there and confirmed. ``retrieval.tag_search``
exists to search inside the tag set before that cut, the same fix
``retrieval.kind_scope`` already made for ``kind``.

Composed path only: a real daemon subprocess on a private port with its own
data folder, saves and recalls over its HTTP API, and the database read back
directly where no HTTP surface exists for the mutation under test (there is
no HTTP route to edit a memory's tags or metadata — only its content).
Synthetic, de-identified fixture text throughout.
"""
from __future__ import annotations

import sqlite3
import uuid

import pytest

from tests.test_integration.test_mcp_declared_kind_transport import _start_daemon
from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    _foreign_daemon_pids,
    _reserve_private_port,
)

RUN = uuid.uuid4().hex[:6].upper()
#: More than any channel's candidate width (semantic 50, BM25 50).
CROWD = 90
QUESTION = f"kestrel checkpoint deployment parameters window {RUN}"


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    root = tmp_path_factory.mktemp("tag-crowd")
    data_root = root / "data"
    data_root.mkdir()
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    foreign = _foreign_daemon_pids()
    d = _start_daemon(data_root, port, root)
    try:
        yield d
    finally:
        d.stop(foreign)


def _save(daemon, content: str, tags: str, key: str) -> dict:
    code, out = daemon.request("POST", "/remember", {
        "content": content, "tags": tags, "idempotency_key": key,
    })
    assert code in (200, 202) and out.get("ok") is True, out
    if code == 202:
        daemon._wait_committed(out["profile"], key)
    return out


def test_the_tagged_memory_behind_a_crowd_of_closer_memories_is_found(daemon) -> None:
    for i in range(CROWD):
        _save(daemon, f"Kestrel checkpoint deployment parameters window {RUN}, "
                     f"synthetic status note {i}.", "status", f"tag-crowd-{RUN}-{i}")
    # The tagged memory's own words never mention the tag's label at all -
    # finding it by content would require the filter, not the words.
    tagged = _save(daemon, f"Kestrel checkpoint {RUN}: the team chose a manual "
                          "approval gate.", "decision-record", f"tag-decision-{RUN}")
    code, unfiltered = daemon.request("GET", "/recall", params={"q": QUESTION, "limit": 50})
    assert code == 200, unfiltered
    code, out = daemon.request(
        "GET", "/recall",
        params={"q": QUESTION, "tags": "decision-record", "limit": 10})
    assert code == 200, out
    shown = {r["fact_id"] for r in out.get("results", [])}
    assert set(tagged["fact_ids"]) & shown, {
        "tagged": tagged["fact_ids"], "shown": sorted(shown),
        "unfiltered_has_it": bool(set(tagged["fact_ids"])
                                 & {r["fact_id"] for r in unfiltered.get("results", [])}),
    }
    assert out["tag_scope"]["applied"] is True
    assert out["tag_scope"]["matched"] >= 1


def test_tags_match_all_versus_any(daemon) -> None:
    both = _save(daemon, f"Halcyon rollout gate {RUN} covers both systems.",
                "alpha,beta", f"tag-both-{RUN}")
    one = _save(daemon, f"Halcyon rollout gate {RUN} covers one system only.",
               "alpha", f"tag-one-{RUN}")
    code, all_out = daemon.request(
        "GET", "/recall",
        params={"q": f"Halcyon rollout gate {RUN}", "tags": "alpha,beta",
                "tags_match": "all", "limit": 10})
    assert code == 200, all_out
    shown_all = {r["fact_id"] for r in all_out["results"]}
    assert set(both["fact_ids"]) & shown_all
    assert not (set(one["fact_ids"]) & shown_all)

    code, any_out = daemon.request(
        "GET", "/recall",
        params={"q": f"Halcyon rollout gate {RUN}", "tags": "alpha,beta",
                "tags_match": "any", "limit": 10})
    assert code == 200, any_out
    shown_any = {r["fact_id"] for r in any_out["results"]}
    assert set(both["fact_ids"]) & shown_any
    assert set(one["fact_ids"]) & shown_any


def test_an_unmatched_filter_says_whether_the_tag_exists_at_all(daemon) -> None:
    _save(daemon, f"Unrelated synthetic fixture note {RUN}.", "unrelated-tag",
         f"tag-unrelated-{RUN}")
    token = f"NOTAG{RUN}"
    code, out = daemon.request(
        "GET", "/recall",
        params={"q": f"fixture {token}", "tags": f"never-stored-{token}", "limit": 10})
    assert code == 200, out
    scope = out["tag_scope"]
    assert scope["applied"] is True
    assert scope["matched"] == 0
    assert scope["reason"] == "no_memory_has_tag"
    assert scope["note"]


def test_deletion_removes_a_tagged_memory_from_the_filter(daemon) -> None:
    saved = _save(daemon, f"Perishable synthetic record {RUN} for deletion.",
                  "perishable-tag", f"tag-delete-{RUN}")
    fact_id = saved["fact_ids"][0]
    code, before = daemon.request(
        "GET", "/recall",
        params={"q": f"perishable synthetic record {RUN}", "tags": "perishable-tag",
                "limit": 10})
    assert code == 200
    assert fact_id in {r["fact_id"] for r in before["results"]}
    code, _ = daemon.request("DELETE", f"/api/memories/{fact_id}")
    assert code in (200, 202, 204), code
    code, after = daemon.request(
        "GET", "/recall",
        params={"q": f"perishable synthetic record {RUN}", "tags": "perishable-tag",
                "limit": 10})
    assert code == 200
    assert fact_id not in {r["fact_id"] for r in after["results"]}


def test_an_edited_tag_is_found_by_the_new_label_only(daemon) -> None:
    saved = _save(daemon, f"Mutable synthetic record {RUN} for an edit.",
                  "old-edit-tag", f"tag-edit-{RUN}")
    fact_id = saved["fact_ids"][0]
    # No HTTP surface edits a memory's tags today (only its content) - the
    # storage layer is the authority this exercises directly, read back from
    # the same file the daemon itself writes.
    conn = sqlite3.connect(daemon.data_root / "memory.db")
    try:
        row = conn.execute(
            "SELECT m.memory_id FROM memories m JOIN atomic_facts f "
            "ON f.memory_id = m.memory_id WHERE f.fact_id = ?", (fact_id,),
        ).fetchone()
        assert row is not None
        conn.execute(
            "UPDATE memories SET metadata_json = json_set(metadata_json, "
            "'$.tags', 'new-edit-tag') WHERE memory_id = ?", (row[0],),
        )
        conn.commit()
    finally:
        conn.close()
    code, old = daemon.request(
        "GET", "/recall",
        params={"q": f"mutable synthetic record {RUN}", "tags": "old-edit-tag",
                "limit": 10})
    assert code == 200
    assert fact_id not in {r["fact_id"] for r in old["results"]}
    code, new = daemon.request(
        "GET", "/recall",
        params={"q": f"mutable synthetic record {RUN}", "tags": "new-edit-tag",
                "limit": 10})
    assert code == 200
    assert fact_id in {r["fact_id"] for r in new["results"]}


def test_another_profiles_tagged_memory_never_appears(daemon) -> None:
    daemon.precreate_profiles(("tag-crowd-other",))
    _save(daemon, f"Other-profile synthetic record {RUN}.", "cross-profile-tag",
         f"tag-other-profile-{RUN}")  # default profile, decoy-named tag
    code, decoy = daemon.request(
        "POST", "/remember",
        {"content": f"Other-profile synthetic record {RUN} v2.",
         "tags": "cross-profile-tag", "idempotency_key": f"tag-other-v2-{RUN}",
         "profile_id": "tag-crowd-other"})
    assert code in (200, 202) and decoy.get("ok") is True, decoy
    if code == 202:
        daemon._wait_committed("tag-crowd-other", f"tag-other-v2-{RUN}")
    code, mine = daemon.request(
        "GET", "/recall",
        params={"q": f"synthetic record {RUN}", "tags": "cross-profile-tag", "limit": 10})
    assert code == 200, mine
    code, theirs = daemon.request(
        "GET", "/recall",
        params={"q": f"synthetic record {RUN}", "tags": "cross-profile-tag",
                "profile_id": "tag-crowd-other", "limit": 10})
    assert code == 200, theirs
    mine_ids = {r["fact_id"] for r in mine["results"]}
    theirs_ids = {r["fact_id"] for r in theirs["results"]}
    assert mine_ids, "the default profile's own tagged memory must be findable"
    assert theirs_ids, "the other profile's own tagged memory must be findable"
    assert not (mine_ids & theirs_ids)
