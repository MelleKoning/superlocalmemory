# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Q9 (2026-10-06): session_init must stay small even on a huge community.

Measured on Varun's real 4.1.21 install: ``session_init(max_results=3)``
returned 104,512 characters. 76,320 of those (73%) were
``thematic_context.member_fact_ids`` — the FULL stored membership of a
3,816-member community, echoed into a response that asked for 3 results.
``RetrievalEngine._community_context`` (src/superlocalmemory/retrieval/
engine.py) now returns a bounded sample instead; this proves the fix holds
through the real ``session_init`` MCP tool, mocking only the network
boundary (``pool_recall``) the way the rest of this test module already does.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from unittest.mock import MagicMock, patch

from superlocalmemory.mcp._pool_adapter import (
    PoolFact,
    PoolRecallItem,
    PoolRecallResponse,
)
from superlocalmemory.mcp.tools_active import register_active_tools

# A real session_init/recall response, even a generous one, should never
# need six figures of characters to answer "what are the top memories".
# 20,000 chars comfortably covers a bounded sample + bounded content and
# would have been blown past ~4x by the old unbounded member list alone.
_SIZE_BUDGET_CHARS = 20_000

#: Member count straight from the Q9 evidence (session_init's own recall).
_HUGE_COMMUNITY_MEMBER_COUNT = 3_816


class _MockServer:
    def __init__(self):
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _session_init_tool():
    server = _MockServer()
    get_engine = MagicMock()
    register_active_tools(server, get_engine)
    return server.tools["session_init"], get_engine


def _huge_community_context(sample_size: int = 8) -> dict:
    """What the FIXED engine now attaches: a sample, never the full list."""
    sample = [f"fact-{i:05d}" for i in range(sample_size)]
    return {
        "community_id": 0,
        "summary": "A large, long-running community of related facts.",
        "keywords": "architecture, decisions, infra",
        "member_fact_ids": sample,
        "member_count": _HUGE_COMMUNITY_MEMBER_COUNT,
        "member_fact_ids_truncated": True,
        "coverage": 0.8,
        "matched_results": sample_size,
    }


def _old_buggy_community_context(sample_size: int = 8) -> dict:
    """What the engine used to attach, for the regression comparison below:
    the ENTIRE stored membership, unbounded."""
    ctx = _huge_community_context(sample_size)
    ctx["member_fact_ids"] = [
        f"fact-{i:05d}" for i in range(_HUGE_COMMUNITY_MEMBER_COUNT)
    ]
    del ctx["member_count"]
    del ctx["member_fact_ids_truncated"]
    return ctx


def _response_with_community_context(ctx: dict, result_count: int = 3) -> PoolRecallResponse:
    results = [
        PoolRecallItem(
            fact=PoolFact(fact_id=f"fact-{i:05d}", content=f"memory {i} content"),
            score=0.9 - i * 0.05,
        )
        for i in range(result_count)
    ]
    metadata = MappingProxyType({"thematic_context": ctx})
    return PoolRecallResponse(results=results, metadata=metadata)


class TestSessionInitSizeBudget:
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_session_init_stays_within_budget_on_a_huge_community(
        self, mock_pool_recall, MockRulesEngine,
    ):
        import asyncio

        rules = MagicMock()
        rules.should_recall.return_value = True
        rules.get_recall_config.return_value = {
            "enabled": True, "relevance_threshold": 0.0,
            "max_memories_injected": 10,
        }
        MockRulesEngine.return_value = rules
        mock_pool_recall.return_value = _response_with_community_context(
            _huge_community_context(),
        )

        session_init, get_engine = _session_init_tool()
        engine = MagicMock()
        engine.profile_id = "default"
        engine.db.get_pinned.return_value = []
        get_engine.return_value = engine

        result = asyncio.run(session_init(project_path="/workspace", max_results=3))

        payload = json.dumps(result, separators=(",", ":"), default=str)
        assert len(payload) < _SIZE_BUDGET_CHARS, (
            f"session_init payload is {len(payload)} chars, over the "
            f"{_SIZE_BUDGET_CHARS}-char budget"
        )
        # The fix, concretely: a bounded sample, the honest count, and an
        # explicit truncation flag — never a silent partial list.
        ctx = result["thematic_context"]
        assert len(ctx["member_fact_ids"]) <= 10
        assert ctx["member_count"] == _HUGE_COMMUNITY_MEMBER_COUNT
        assert ctx["member_fact_ids_truncated"] is True

    def test_the_old_unbounded_shape_would_have_blown_the_budget(self):
        """Anchors the regression: this is what 104KB looked like."""
        huge_ctx = _old_buggy_community_context()
        payload = json.dumps(
            {"thematic_context": huge_ctx}, separators=(",", ":"), default=str,
        )
        assert len(payload) > _SIZE_BUDGET_CHARS * 2, (
            "the old unbounded member_fact_ids shape no longer reproduces "
            "the oversized payload this test exists to catch"
        )
