# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Credentials are kept in SLM, and never sent to a service on another machine.

SLM stores a memory exactly as written. Where memory text is sent to a hosted
model or a cloud embedding service, every credential is replaced first. A
model or embedder running on this machine sees the text as it is.
"""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.config import EmbeddingConfig, LLMConfig
from superlocalmemory.core.outbound_redaction import for_endpoint, is_local_endpoint
from superlocalmemory.llm.backbone import LLMBackbone

_KEY = "AKIAIOSFODNN7EXAMPLE"
_TEXT = f"deploy uses the AWS key {_KEY} from the vault"


@pytest.mark.parametrize("url", [
    "http://localhost:11434/api/chat",
    "http://127.0.0.1:8080/v1",
    "http://[::1]:9000/v1/embeddings",
    "http://my-box.localhost:1234",
])
def test_an_endpoint_on_this_machine_is_local(url: str) -> None:
    assert is_local_endpoint(url)


@pytest.mark.parametrize("url", [
    "", "https://api.openai.com/v1/chat/completions", "https://x.openai.azure.com",
    "http://192.168.1.20:11434/api/chat", "https://localhost.evil.example/v1",
    "not a url",
])
def test_anything_else_is_remote(url: str) -> None:
    assert not is_local_endpoint(url)


def test_text_for_a_remote_endpoint_loses_the_credential_and_keeps_the_rest() -> None:
    out = for_endpoint(_TEXT, "https://api.openai.com/v1")
    assert _KEY not in out
    assert "deploy uses the AWS key" in out and "from the vault" in out


def test_text_for_a_local_endpoint_is_unchanged() -> None:
    assert for_endpoint(_TEXT, "http://localhost:11434") == _TEXT


def _capture_backbone(monkeypatch, provider: str, api_base: str) -> list[dict]:
    sent: list[dict] = []
    backbone = LLMBackbone(LLMConfig(provider=provider, model="m", api_key="k",
                                     api_base=api_base))

    def _send(url, headers, payload):
        sent.append({"url": url, "payload": payload})
        return {"choices": [{"message": {"content": "ok"}}],
                "message": {"content": "ok"},
                "content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(backbone, "_send", _send)
    backbone.generate(prompt=_TEXT, system=f"system note {_KEY}")
    return sent


@pytest.mark.parametrize("provider,api_base", [
    ("openai", "https://api.openai.com/v1"),
    ("anthropic", ""),
    ("azure", "https://x.openai.azure.com"),
    ("openrouter", ""),
])
def test_a_hosted_model_never_receives_a_credential(monkeypatch, provider, api_base) -> None:
    sent = _capture_backbone(monkeypatch, provider, api_base)
    assert sent, "the request was not made"
    assert _KEY not in json.dumps(sent[0]["payload"])


def test_a_model_on_this_machine_sees_the_text_as_written(monkeypatch) -> None:
    sent = _capture_backbone(monkeypatch, "ollama", "http://localhost:11434")
    assert _KEY in json.dumps(sent[0]["payload"])


class _Response:
    status_code = 200

    def __init__(self, n: int) -> None:
        self._n = n

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"data": [{"embedding": [0.0, 1.0], "index": i} for i in range(self._n)]}


class _Client:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None, **_):  # noqa: A002
        self.bodies.append(json)
        return _Response(len(json["input"]))


def _embedder(monkeypatch, **cfg):
    from superlocalmemory.core.embeddings import EmbeddingService

    service = EmbeddingService(EmbeddingConfig(dimension=2, **cfg))
    client = _Client()
    monkeypatch.setattr(service, "_get_http_client", lambda: client)
    return service, client


def test_a_cloud_embedder_never_receives_a_credential(monkeypatch) -> None:
    service, client = _embedder(monkeypatch, provider="openai",
                                api_endpoint="https://api.openai.com/v1", api_key="k")
    service._openai_compatible_embed_batch([_TEXT])
    assert client.bodies and _KEY not in json.dumps(client.bodies)


def test_an_azure_embedder_never_receives_a_credential(monkeypatch) -> None:
    service, client = _embedder(monkeypatch, provider="cloud",
                                api_endpoint="https://x.openai.azure.com", api_key="k",
                                deployment_name="d")
    service._cloud_embed_batch([_TEXT])
    assert client.bodies and _KEY not in json.dumps(client.bodies)


def test_an_embedder_on_this_machine_sees_the_text_as_written(monkeypatch) -> None:
    service, client = _embedder(monkeypatch, provider="openai",
                                api_endpoint="http://localhost:8080/v1", api_key="")
    service._openai_compatible_embed_batch([_TEXT])
    assert _KEY in json.dumps(client.bodies)
