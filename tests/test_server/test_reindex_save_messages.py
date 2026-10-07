# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""What a save says about an embedding change matches what actually happens.

Since 4.1.22 a model change is a background re-index run by the daemon:
recall keeps using the old model and its vectors until the switch completes.
The save routes still said "re-indexing will run on next recall" (MCP
set_mode) or flagged ``needs_reindex`` from a bare name comparison with no
explanation at all (PUT /embedding/config, POST /mode/set) — true for no
version of the product since the background job landed, and wrong for an
alias of the live model, which needs no re-index.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

STALE = ("next recall", "next use")


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr("superlocalmemory.core.config.DEFAULT_BASE_DIR", tmp_path)
    monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({
        "mode": "a",
        "embedding": {"provider": "openai", "api_endpoint": "http://127.0.0.1:9/v1",
                      "model_name": "old-model", "dimension": 768, "api_key": ""},
    }), encoding="utf-8")
    return tmp_path


def _request(body: dict, runner=None):
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    request.app.state = SimpleNamespace(embedding_reindex=runner) if runner else MagicMock(spec=[])
    request.client = SimpleNamespace(host="127.0.0.1")
    return request


def _call(route, body, runner=None):
    async def _no_apply(*_a, **_k):
        return None
    with patch("superlocalmemory.server.routes.v3_api._apply_runtime_config", _no_apply), \
            patch("superlocalmemory.server.rbac_enforce.require_manage", lambda _r: None):
        response = asyncio.run(route(_request(body, runner)))
    if hasattr(response, "body"):
        return response.status_code, json.loads(response.body)
    return 200, response


def _assert_honest(payload: dict, new: str, old: str) -> None:
    assert payload["needs_reindex"] is True, payload
    message = payload["message"]
    assert "background" in message and new in message, message
    assert f"keeps using {old}" in message, message
    assert not any(s in message for s in STALE), message


class TestEmbeddingConfigSave:
    def test_a_model_change_without_the_daemon_runner_says_what_happens(self, base):
        from superlocalmemory.server.routes.v3_api import set_embedding_config

        code, payload = _call(set_embedding_config, {"model_name": "new-model", "dimension": 384})
        assert code == 200
        _assert_honest(payload, "new-model", "old-model")

    def test_an_unchanged_space_needs_no_reindex(self, base):
        from superlocalmemory.server.routes.v3_api import set_embedding_config

        code, payload = _call(set_embedding_config, {"api_key": "rotated"})
        assert code == 200
        assert payload["needs_reindex"] is False, payload
        assert payload["message"] == ""

    def test_with_the_runner_a_200_never_claims_a_pending_reindex(self, base):
        """The runner answers 202 for a real switch; a 200 (here: the runner
        reports the store already holds that model) has nothing to re-index."""
        from superlocalmemory.core.embedding_reindex import NoChange
        from superlocalmemory.server.routes.v3_api import set_embedding_config

        runner = MagicMock()
        runner.request_switch.side_effect = NoChange("already the embedding model")
        code, payload = _call(set_embedding_config,
                              {"model_name": "new-model", "dimension": 384}, runner)
        assert code == 200
        assert payload["needs_reindex"] is False, payload
        assert payload["message"] == ""


class TestModeSetSave:
    def test_a_model_change_without_the_daemon_runner_says_what_happens(self, base):
        from superlocalmemory.server.routes.v3_api import set_full_config

        code, payload = _call(set_full_config, {
            "mode": "a", "embedding_provider": "openai",
            "embedding_endpoint": "http://127.0.0.1:9/v1",
            "embedding_model": "new-model", "embedding_dimension": 384})
        assert code == 200, payload
        _assert_honest(payload, "new-model", "old-model")


class TestModeQuickSwitch:
    def test_a_quick_switch_keeps_the_model_and_says_no_reindex(self, base):
        from superlocalmemory.server.routes.v3_api import set_mode

        with patch("superlocalmemory.server.routes.helpers.log_mode_change", lambda *a, **k: None):
            code, payload = _call(set_mode, {"mode": "b"})
        assert code == 200, payload
        assert payload["needs_reindex"] is False
        assert payload["message"] == ""
        saved = json.loads((base / "config.json").read_text(encoding="utf-8"))
        assert saved["embedding"]["model_name"] == "old-model"


class TestMcpSetMode:
    def test_a_mode_file_naming_another_model_says_what_happens(self, base):
        from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
        from superlocalmemory.mcp.tools_v3 import register_v3_tools
        from superlocalmemory.storage.models import Mode

        tools: dict = {}

        class _Server:
            def tool(self, *a, **k):
                def register(fn):
                    tools[fn.__name__] = fn
                    return fn
                return register

        engine = MagicMock()
        engine.profile_id = "default"
        register_v3_tools(_Server(), MagicMock(return_value=engine))
        old = SLMConfig.load()
        new = SLMConfig.load()
        new.mode = Mode.B
        new.embedding = EmbeddingConfig(provider="openai", api_endpoint="http://127.0.0.1:9/v1",
                                        model_name="new-model", dimension=384)
        with patch("superlocalmemory.mcp.tools_v3.authorize_mcp_mutation", MagicMock()), \
                patch("superlocalmemory.core.config.SLMConfig.load", return_value=old), \
                patch("superlocalmemory.core.config.SLMConfig.switch_mode", return_value=new), \
                patch("superlocalmemory.mcp.server.reset_engine", lambda: None):
            payload = asyncio.run(tools["set_mode"]("b"))
        assert payload["success"] is True, payload
        _assert_honest(payload, "new-model", "old-model")


@pytest.mark.parametrize("rel", [
    "src/superlocalmemory/server/routes/v3_api.py",
    "src/superlocalmemory/mcp/tools_v3.py",
    "src/superlocalmemory/ui/js/auto-settings.js",
])
def test_no_save_path_promises_a_reindex_on_next_recall(rel):
    src = (Path(__file__).resolve().parents[2] / rel).read_text(encoding="utf-8")
    for line in src.splitlines():
        low = line.lower()
        if "re-index" in low or "reindex" in low or "re-embed" in low:
            assert not any(s in low for s in STALE), f"{rel}: {line.strip()}"
