# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""An unreadable time window is refused on every door, never ignored (R5).

MCP and HTTP passed ``window`` through raw. The engine could not parse it and
applied no filter, so "lastweek" (a typo for "7d") returned every memory with
no hint that the filter was dropped. The CLI already refused it. Every door now
refuses it the same way: ``INVALID_TIME_FILTER`` (HTTP 400), before retrieval.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.retrieval.time_filter import (
    CODE,
    InvalidTimeFilter,
    check_window,
    first_invalid,
)

BAD = "lastweek"


class TestTheCheck:
    @pytest.mark.parametrize("ok", ["7d", " 24h ", "2026-07-01..2026-07-31",
                                    "2026-07-01,2026-07-31",
                                    ("2026-07-01", "2026-07-31")])
    def test_readable_windows_pass(self, ok) -> None:
        assert check_window(ok) is not None

    @pytest.mark.parametrize("blank", [None, "", "   ", ()])
    def test_blank_is_no_filter(self, blank) -> None:
        assert check_window(blank) is None

    @pytest.mark.parametrize("bad", [BAD, "7 days ago", "2026-13-01..2026-13-02",
                                     ("2026-07-01",), 7])
    def test_unreadable_windows_raise(self, bad) -> None:
        with pytest.raises(InvalidTimeFilter) as info:
            check_window(bad)
        assert info.value.as_dict()["code"] == CODE
        assert info.value.field == "window"

    def test_first_invalid_names_the_field(self) -> None:
        assert first_invalid({"window": "7d", "as_of": ""}) is None
        assert first_invalid({"window": "7d", "as_of": "nope"}).field == "as_of"
        assert first_invalid({"window": BAD}).field == "window"


def test_the_engine_refuses_rather_than_running_unfiltered(engine_with_mock_deps) -> None:
    with pytest.raises(InvalidTimeFilter):
        engine_with_mock_deps.recall("anything", window=BAD)


# -- MCP ------------------------------------------------------------------------


class _Server:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def test_mcp_refuses_before_asking_the_daemon() -> None:
    from superlocalmemory.mcp.tools_core import register_core_tools

    srv = _Server()
    register_core_tools(srv, MagicMock())
    pool = MagicMock()
    with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool):
        out = asyncio.run(srv.tools["recall"](query="what changed", window=BAD))
    assert out["success"] is False
    assert out["code"] == CODE and out["retryable"] is False
    assert BAD in out["error"]
    pool.recall.assert_not_called()


# -- HTTP -----------------------------------------------------------------------


def _client(engine) -> TestClient:
    from superlocalmemory.server.unified_daemon import create_app

    calls: list = []

    def recall(*a, **k):
        calls.append(k)
        return SimpleNamespace(results=[], query="q", query_type="lookup",
                               retrieval_time_ms=1.0, channel_weights={},
                               total_candidates=0, no_confident_match=True)

    engine.recall = recall
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    client = TestClient(app)
    client.calls = calls
    return client


def test_http_get_recall_is_400(engine_with_mock_deps) -> None:
    client = _client(engine_with_mock_deps)
    r = client.get("/recall", params={"q": "what changed", "window": BAD})
    assert r.status_code == 400, r.text
    assert r.json()["code"] == CODE and r.json()["field"] == "window"
    assert client.calls == []


def test_http_dashboard_search_is_400(engine_with_mock_deps) -> None:
    client = _client(engine_with_mock_deps)
    r = client.post("/api/search", json={"query": "what changed", "limit": 5,
                                         "window": BAD})
    assert r.status_code == 400, r.text
    assert r.json()["code"] == CODE
    assert client.calls == []


def test_http_dashboard_search_date_range_is_checked_too(engine_with_mock_deps) -> None:
    client = _client(engine_with_mock_deps)
    r = client.post("/api/search", json={"query": "what changed", "limit": 5,
                                         "date_from": "yesterday", "date_to": "today"})
    assert r.status_code == 400, r.text
    assert client.calls == []


def test_http_recall_trace_is_400() -> None:
    from starlette.responses import JSONResponse

    from superlocalmemory.server.routes.v3_api import recall_trace

    class _Req:
        app = SimpleNamespace(state=SimpleNamespace())
        headers: dict = {}
        cookies: dict = {}

        async def json(self):
            return {"query": "what changed", "window": BAD}

    response = asyncio.run(recall_trace(_Req()))
    assert isinstance(response, JSONResponse) and response.status_code == 400
    assert json.loads(response.body)["code"] == CODE
