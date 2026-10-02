# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every recall metadata field the HTTP API returns, MCP returns too.

The daemon builds its recall envelope with ``recall_response_metadata``. The
MCP ``recall`` and ``recall_trace`` tools and the pool adapter used to copy a
hand-picked subset, so an agent could not see who chose the final order
(``reranker_status``), whether a channel was abandoned, or the temporal
frame. The forward is generic: a field added to ``recall_response_metadata``
later reaches MCP with no other change — the last test proves it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.server import recall_serializer


class _CaptureServer:
    def __init__(self):
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator

    def resource(self, *args, **kwargs):
        return lambda fn: fn

    def prompt(self, *args, **kwargs):
        return lambda fn: fn


def _engine_response():
    """A RecallResponse-shaped object with a non-default value in every field."""
    return SimpleNamespace(
        results=[SimpleNamespace(fact=SimpleNamespace(created_at="2026-09-01T10:00:00+00:00"))],
        score_contract_version="2",
        calibration_status="measured_synthetic_set_not_calibrated",
        calibration_id="jev:typesafe:model:sufficiency-v1:top3",
        query_id="q-123",
        answer_confidence=0.81,
        abstained=False,
        abstention_reason=None,
        community_context="Team decisions about auth",
        incomplete_channels=("hopfield",),
        channel_status={"semantic": "ok", "hopfield": "timed_out"},
        reranker_status="jev_listwise",
        local_reranker_status="cross_encoder",
    )


def _daemon_payload(metadata: dict) -> dict:
    return {
        "ok": True, "profile": "default", "profile_generation": 1, "query": "q",
        "query_type": "factual", "result_count": 0, "retrieval_time_ms": 12.0,
        "channel_weights": {}, "total_candidates": 0, "results": [], "count": 0,
        "no_confident_match": False, **metadata,
    }


class _FakePool:
    def __init__(self, payload):
        self.payload = payload

    def recall(self, *args, **kwargs):
        return dict(self.payload)


def _tool(name: str, monkeypatch, payload: dict):
    from superlocalmemory.mcp import _daemon_proxy

    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: _FakePool(payload))
    srv = _CaptureServer()
    if name == "recall":
        from superlocalmemory.mcp.tools_core import register_core_tools
        register_core_tools(srv, lambda: SimpleNamespace(profile_id="default"))
    else:
        from superlocalmemory.mcp.tools_v3 import register_v3_tools
        register_v3_tools(srv, lambda: SimpleNamespace(profile_id="default"))
    return srv.tools[name]


@pytest.mark.parametrize("name", ["recall", "recall_trace"])
def test_every_metadata_key_reaches_the_mcp_tool(name, monkeypatch):
    metadata = recall_serializer.recall_response_metadata(_engine_response())
    tool = _tool(name, monkeypatch, _daemon_payload(metadata))
    out = asyncio.run(tool(query="q"))
    assert out["success"] is True, out
    missing = {k for k in metadata if k not in out}
    assert not missing, f"{name} drops {sorted(missing)}"
    assert {k: out[k] for k in metadata} == metadata


def test_every_metadata_key_reaches_the_pool_adapter(monkeypatch):
    from superlocalmemory.mcp import _pool_adapter

    metadata = recall_serializer.recall_response_metadata(_engine_response())
    monkeypatch.setattr(_pool_adapter, "_pool", lambda: _FakePool(_daemon_payload(metadata)))
    response = _pool_adapter.pool_recall("q")
    assert dict(response.metadata) == metadata
    assert response.calibration_status == metadata["calibration_status"]


@pytest.mark.parametrize("name", ["recall", "recall_trace"])
def test_a_metadata_field_added_later_reaches_mcp_with_no_other_change(name, monkeypatch):
    real = recall_serializer.recall_response_metadata

    def with_a_new_field(response):
        return {**real(response), "answer_check_status": getattr(
            response, "answer_check_status", "off")}

    monkeypatch.setattr(recall_serializer, "recall_response_metadata", with_a_new_field)
    response = _engine_response()
    response.answer_check_status = "judged"
    payload = _daemon_payload(with_a_new_field(response))
    out = asyncio.run(_tool(name, monkeypatch, payload)(query="q"))
    assert out["answer_check_status"] == "judged"


def test_an_older_daemon_without_the_new_fields_gets_their_defaults(monkeypatch):
    from superlocalmemory.mcp import _pool_adapter

    legacy = {"ok": True, "results": [], "calibration_status": "uncalibrated"}
    monkeypatch.setattr(_pool_adapter, "_pool", lambda: _FakePool(legacy))
    response = _pool_adapter.pool_recall("q")
    assert response.metadata["reranker_status"] == "not_configured"
    assert response.metadata["incomplete_channels"] == []
