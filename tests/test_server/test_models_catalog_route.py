# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""GET /api/v3/models/catalog: the catalogue and what to pick on this machine."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import config_api


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config_api, "MEMORY_DIR", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "llm": {"provider": "ollama", "model": "llama3.2",
                "base_url": "http://localhost:11434"},
    }), encoding="utf-8")
    monkeypatch.setattr("superlocalmemory.core.machine.total_ram_gb", lambda: 24.0)
    app = FastAPI()
    app.include_router(config_api.router)
    return TestClient(app)


def test_installed_models_are_ranked_and_the_best_is_named(client, monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(200, {"models": [
        {"name": "llama3.2:latest", "size": 1}, {"name": "gemma3:4b", "size": 1},
        {"name": "nomic-embed-text:latest", "size": 1}]}))
    body = client.get("/api/v3/models/catalog").json()
    assert body["machine"] == {"ram_gb": 24.0}
    assert body["ollama"]["reachable"] is True
    assert body["best_local_llm"] == "gemma3:4b"
    ranked = [r["model"] for r in body["local_recommendations"] if r["installed"]]
    assert ranked == ["gemma3:4b", "llama3.2"]
    assert body["defaults"]["local_llm"] == "gemma3:4b"
    assert {e["id"] for e in body["hosted_llms"]} >= {body["defaults"]["hosted_llm"]}


def test_without_ollama_it_still_answers_with_pull_suggestions(client, monkeypatch) -> None:
    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    body = client.get("/api/v3/models/catalog").json()
    assert body["ollama"]["reachable"] is False and "ollama serve" in body["ollama"]["detail"]
    assert body["local_recommendations"]
    assert all(not r["installed"] for r in body["local_recommendations"])
    assert body["best_local_llm"] == body["defaults"]["local_llm"]
