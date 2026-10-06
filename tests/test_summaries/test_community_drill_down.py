# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``get_memory_summary(kind="community")`` — the Q9 explicit drill-down.

``thematic_context`` on recall/session_init now carries only a small,
relevance-bounded SAMPLE of a community's member ids
(``RetrievalEngine._community_context``). This is where the rest lives: the
full, paged membership, read on demand, never on the recall hot path.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from superlocalmemory.mcp import tools_summaries


def _row(community_id: int = 0, member_count: int = 4000) -> dict:
    members = [f"m{i}" for i in range(member_count)]
    return {
        "community_id": community_id,
        "summary": "A very large community.",
        "keywords": "big, community",
        "fact_ids_json": json.dumps(members),
        "fact_count": member_count,
    }


@pytest.fixture()
def tool():
    db = MagicMock()
    db.execute.return_value = [_row()]
    tools: dict = {}

    class Collector:
        def tool(self, *a, **k):
            def register(fn):
                tools[fn.__name__] = fn
                return fn
            return register

    tools_summaries.register_summary_tools(
        Collector(),
        lambda: SimpleNamespace(profile_id="default", config=None, db=db),
    )
    return tools["get_memory_summary"], db


class TestCommunityDrillDown:
    def test_default_page_is_bounded_not_everything(self, tool) -> None:
        fn, _db = tool
        answer = asyncio.run(fn(kind="community", target="0"))
        assert answer["success"] is True
        assert answer["kind"] == "community"
        assert len(answer["source_fact_ids"]) == tools_summaries._DEFAULT_COMMUNITY_MEMBER_LIMIT
        assert answer["source_fact_ids"][0] == "m0"
        assert answer["metadata"]["member_count"] == 4000
        assert answer["metadata"]["has_more"] is True

    def test_explicit_limit_and_offset_page_through_it(self, tool) -> None:
        fn, _db = tool
        answer = asyncio.run(fn(kind="community", target="0", limit=10, offset=4000 - 5))
        assert answer["success"] is True
        assert answer["source_fact_ids"] == ["m3995", "m3996", "m3997", "m3998", "m3999"]
        assert answer["metadata"]["has_more"] is False

    def test_unknown_community_is_refused_not_empty(self, tool) -> None:
        fn, db = tool
        db.execute.return_value = []  # no row for this profile/community
        answer = asyncio.run(fn(kind="community", target="99"))
        assert answer["success"] is False
        assert "99" in answer["error"]

    def test_non_numeric_target_is_refused(self, tool) -> None:
        fn, _db = tool
        answer = asyncio.run(fn(kind="community", target="not-a-number"))
        assert answer["success"] is False
        assert "community_id" in answer["error"]

    def test_unknown_kind_still_lists_community(self, tool) -> None:
        fn, _db = tool
        answer = asyncio.run(fn(kind="bogus"))
        assert answer["success"] is False
        assert "community" in answer["error"]


class TestCommunityDrillDownHelperDirect:
    def test_no_engine_db_returns_none(self) -> None:
        engine = SimpleNamespace(profile_id="default")  # no .db / ._db
        result = tools_summaries._community_drill_down(engine, "default", 0, 0, 0)
        assert result is None

    def test_negative_offset_clamps_to_zero(self) -> None:
        db = MagicMock()
        db.execute.return_value = [_row(member_count=3)]
        engine = SimpleNamespace(db=db)
        result = tools_summaries._community_drill_down(engine, "default", 0, 0, -5)
        assert result is not None
        assert result.source_fact_ids == ["m0", "m1", "m2"]
