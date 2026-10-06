# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Every path that sends memory text off this machine is screened.

Each test drives one real exit with a memory that holds a credential and
captures what would go on the wire. Nothing is sent: the HTTP clients are
patched to record the request and stop.

The exits covered are the ones that used to skip the screen: the cluster
summarizer (which posted to a hard-coded OpenRouter with whatever key it
found), the consolidation and entity-compiler fallbacks (which posted
Ollama-format requests to the cloud host), a LAN Ollama embedder or
summarizer, the dashboard chat, the session/daily/project summaries, the
evolution prompts and the remote reranker.
"""

from __future__ import annotations

import json
import types
import urllib.request

import httpx
import pytest

_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"  # fake, test-only
_MEMORY = f"Prod OpenAI key is {_KEY} and the db password=Hunt3rTwo!x9"
_AZURE_KEY = "azkey-" + "0123456789abcdef0123456789abcdef"  # fake
_OPENROUTER_KEY = "sk-or-v1-" + "f" * 64  # fake
_AZURE_BASE = "https://contoso.openai.azure.com"
_LAN = "http://192.168.1.50:11434"


class _Stop(Exception):
    """Raised by every fake client after it records the request."""


@pytest.fixture()
def wire(monkeypatch):
    """Record every request any HTTP client in the process would send."""
    sent: list[dict] = []

    def _record(via, url, headers, body):
        sent.append({"via": via, "url": str(url), "headers": dict(headers or {}),
                     "body": body if isinstance(body, str) else json.dumps(body)})
        raise _Stop()

    def _httpx_post(url, *a, **kw):
        _record("httpx.post", url, kw.get("headers"), kw.get("json", kw.get("content")))

    def _client_post(self, url, *a, **kw):
        _record("httpx.Client.post", url, kw.get("headers"),
                kw.get("json", kw.get("content")))

    def _urlopen(req, *a, **kw):
        data = req.data.decode() if getattr(req, "data", None) else ""
        _record("urlopen", req.full_url, dict(req.header_items()), data)

    monkeypatch.setattr(httpx, "post", _httpx_post)
    monkeypatch.setattr(httpx.Client, "post", _client_post)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    # The gate opens through its own no-redirect opener, not the module-level
    # urlopen, so record at the opener too.
    monkeypatch.setattr(urllib.request.OpenerDirector, "open",
                        lambda self, req, *a, **kw: _urlopen(req))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import superlocalmemory.llm.backbone as backbone
    monkeypatch.setattr(backbone.time, "sleep", lambda _s: None)
    return sent


def _cfg(mode: str, provider: str, api_base: str, api_key: str = "", model: str = "m"):
    return types.SimpleNamespace(
        mode=types.SimpleNamespace(value=mode),
        llm=types.SimpleNamespace(provider=provider, api_key=api_key, model=model,
                                  api_base=api_base, timeout_seconds=30,
                                  temperature=0.0, max_tokens=256),
        entity_compilation_enabled=True,
    )


def _llm_cfg(mode: str, provider: str, api_base: str, api_key: str = ""):
    from superlocalmemory.core.config import LLMConfig

    return types.SimpleNamespace(
        mode=types.SimpleNamespace(value=mode),
        llm=LLMConfig(provider=provider, model="gpt-4o", api_key=api_key, api_base=api_base),
        entity_compilation_enabled=True,
    )


def _assert_screened(sent: list[dict]) -> None:
    assert sent, "no request was made"
    for req in sent:
        assert _KEY not in req["body"], f"credential sent to {req['url']}"
        assert "Hunt3rTwo" not in req["body"], f"password sent to {req['url']}"


# -- L2-01: the summarizer uses the configured provider and nothing else ------


def test_mode_c_summary_goes_only_to_the_configured_provider(wire, monkeypatch) -> None:
    from superlocalmemory.core.summarizer import Summarizer

    monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_KEY)
    cfg = _llm_cfg("c", "azure", _AZURE_BASE, _AZURE_KEY)
    Summarizer(cfg).summarize_cluster([{"content": _MEMORY}])
    Summarizer(cfg).synthesize_answer("what is the prod key?", [{"content": _MEMORY}])

    _assert_screened(wire)
    for req in wire:
        assert httpx.URL(req["url"]).host == "contoso.openai.azure.com"
        assert "openrouter" not in req["url"]
        blob = json.dumps(req)
        assert _OPENROUTER_KEY not in blob, "an OpenRouter key went to another host"


def test_an_azure_key_never_goes_to_openrouter(wire) -> None:
    from superlocalmemory.core.summarizer import Summarizer

    Summarizer(_llm_cfg("c", "azure", _AZURE_BASE, _AZURE_KEY)).summarize_cluster(
        [{"content": _MEMORY}])
    for req in wire:
        if _AZURE_KEY in json.dumps(req["headers"]):
            assert httpx.URL(req["url"]).host == "contoso.openai.azure.com"


def test_mode_b_summary_to_a_lan_ollama_is_screened(wire, monkeypatch) -> None:
    from superlocalmemory.core.summarizer import Summarizer
    from superlocalmemory.llm import ollama_reachability

    # This test is about what leaves the machine once a request is actually
    # sent, not about whether this particular LAN address answers from
    # wherever the suite happens to run -- a real user's LAN Ollama IS
    # reachable, and is exactly the case the screen below must still catch.
    monkeypatch.setattr(ollama_reachability, "ollama_reachable", lambda *a, **k: True)
    Summarizer(_llm_cfg("b", "ollama", _LAN)).summarize_cluster([{"content": _MEMORY}])
    _assert_screened(wire)
    assert all(httpx.URL(r["url"]).host == "192.168.1.50" for r in wire)


def test_mode_b_summary_to_this_machine_is_sent_as_written(wire) -> None:
    from superlocalmemory.core.summarizer import Summarizer

    Summarizer(_llm_cfg("b", "ollama", "http://localhost:11434")).summarize_cluster(
        [{"content": _MEMORY}])
    assert wire and _KEY in wire[0]["body"]


def test_a_mode_c_summary_without_a_provider_sends_nothing(wire, monkeypatch) -> None:
    from superlocalmemory.core.summarizer import Summarizer

    monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_KEY)
    out = Summarizer(_llm_cfg("c", "", "")).summarize_cluster([{"content": "Atlas ships."}])
    assert out and wire == []


# -- L2-02: consolidation and the entity compiler in Mode C ---------------------


def test_mode_c_consolidation_never_posts_ollama_format_to_the_cloud(wire) -> None:
    from superlocalmemory.core.fact_consolidator import _generate_summary

    facts = [{"content": _MEMORY}] * 4
    _summary, generated_by = _generate_summary(
        "OpenAI", facts, _llm_cfg("c", "azure", _AZURE_BASE, _AZURE_KEY))
    assert generated_by == "extractive"
    _assert_screened(wire)
    assert not any(r["url"].endswith("/api/generate") for r in wire)


def test_mode_b_consolidation_to_a_lan_ollama_is_screened(wire) -> None:
    from superlocalmemory.core.fact_consolidator import _summarize_with_ollama

    _summarize_with_ollama("OpenAI", [{"content": _MEMORY}] * 4, _cfg("b", "ollama", _LAN))
    _assert_screened(wire)


def test_mode_c_entity_compiler_uses_the_configured_provider(wire) -> None:
    from superlocalmemory.learning.entity_compiler import EntityCompiler

    ec = EntityCompiler(":memory:", _llm_cfg("c", "azure", _AZURE_BASE, _AZURE_KEY))
    assert ec._compile_with_llm("OpenAI", [{"content": _MEMORY}] * 4) is None
    _assert_screened(wire)
    for req in wire:
        assert httpx.URL(req["url"]).host == "contoso.openai.azure.com"
        assert not req["url"].endswith("/api/generate")


def test_mode_b_entity_compiler_to_a_lan_ollama_is_screened(wire) -> None:
    from superlocalmemory.learning.entity_compiler import EntityCompiler

    ec = EntityCompiler(":memory:", _cfg("b", "ollama", _LAN))
    ec._compile_with_llm("OpenAI", [{"content": _MEMORY}] * 4)
    _assert_screened(wire)


# -- L2-03: a LAN Ollama embedder -------------------------------------------------


def _ollama_embedder(base_url: str):
    from superlocalmemory.core.ollama_embedder import OllamaEmbedder

    emb = OllamaEmbedder.__new__(OllamaEmbedder)
    emb._base_url = base_url
    emb._model = "nomic-embed-text"
    emb._dimension = 768
    return emb


def test_a_lan_ollama_embedder_never_receives_a_credential(wire) -> None:
    emb = _ollama_embedder(_LAN)
    with pytest.raises(_Stop):
        emb._call_ollama_embed_batch([_MEMORY])
    with pytest.raises(_Stop):
        emb._call_ollama_embed(_MEMORY)
    _assert_screened(wire)


def test_an_ollama_embedder_on_this_machine_sees_the_text(wire) -> None:
    with pytest.raises(_Stop):
        _ollama_embedder("http://127.0.0.1:11434")._call_ollama_embed_batch([_MEMORY])
    assert _KEY in wire[0]["body"]


# -- summaries (session, daily, project) ----------------------------------------


@pytest.mark.parametrize("module_name", ["session_summary", "daily_reflection"])
def test_a_lan_ollama_summary_is_screened(wire, module_name: str) -> None:
    import importlib

    mod = importlib.import_module(f"superlocalmemory.summaries.{module_name}")
    mod._call_ollama("Summarise.", [{"content": _MEMORY}], _cfg("b", "ollama", _LAN))
    _assert_screened(wire)


def test_a_lan_ollama_project_log_is_screened(wire) -> None:
    from superlocalmemory.summaries import project_work_log as mod

    mod._call_ollama("Summarise.", [{"tool_name": "Bash", "event_type": "x"}],
                     [{"content": _MEMORY}], _cfg("b", "ollama", _LAN))
    _assert_screened(wire)


@pytest.mark.parametrize("module_name", ["session_summary", "daily_reflection",
                                         "project_work_log"])
def test_a_mode_c_summary_never_posts_ollama_format_to_the_cloud(
    wire, module_name: str,
) -> None:
    import importlib
    import inspect

    mod = importlib.import_module(f"superlocalmemory.summaries.{module_name}")
    params = inspect.signature(mod._try_llm).parameters
    cfg = _llm_cfg("c", "azure", _AZURE_BASE, _AZURE_KEY)
    facts = [{"content": _MEMORY}]
    if "tool_rows" in params:
        mod._try_llm("/tmp/p", [], facts, cfg, "c")
    elif "date_str" in params:
        mod._try_llm("2026-10-03", facts, cfg, "c")
    else:
        mod._try_llm("Summarise.", facts, cfg, "c")
    _assert_screened(wire)
    assert not any(r["url"].endswith("/api/generate") for r in wire)


# -- the dashboard chat ----------------------------------------------------------


def _drain_chat(gen) -> None:
    import asyncio

    async def _run():
        try:
            async for _ in gen:
                pass
        except _Stop:
            pass

    asyncio.run(_run())


def test_dashboard_chat_to_a_cloud_provider_is_screened(monkeypatch) -> None:
    from superlocalmemory.core import outbound_http
    from superlocalmemory.server.routes import chat

    bodies: list[bytes] = []

    async def _handler(request):
        bodies.append(request.content)
        return httpx.Response(200, text="data: [DONE]\n")

    real = outbound_http.astream_json

    def _with_mock(method, url, payload, **kw):
        return real(method, url, payload, transport=httpx.MockTransport(_handler), **kw)

    monkeypatch.setattr(outbound_http, "astream_json", _with_mock)
    messages = [{"role": "user", "content": f"Question: {_MEMORY}"}]
    _drain_chat(chat._stream_openai_compat(messages, "gpt-4o", "k", "", "openai"))
    _drain_chat(chat._stream_ollama(messages, "llama3.2", _LAN))
    assert len(bodies) == 2
    for body in bodies:
        assert _KEY.encode() not in body


# -- evolution and the remote reranker keep no tail ------------------------------


def test_an_evolution_prompt_keeps_no_part_of_a_credential(tmp_path, monkeypatch) -> None:
    from superlocalmemory.evolution import llm_dispatch

    seen: list[str] = []
    monkeypatch.setattr(llm_dispatch, "_actual_llm_call",
                        lambda prompt, **kw: seen.append(prompt) or "OK")
    llm_dispatch._dispatch_llm(f"skill note: {_MEMORY}", model="claude-haiku-4-5",
                               learning_db=tmp_path / "learning.db", profile_id="default")
    assert seen and _KEY not in seen[0] and "Hunt3rTwo" not in seen[0]
    assert "[REDACTED:" not in seen[0]
    assert "S9t0" not in seen[0]


def test_remote_rerank_text_keeps_no_part_of_a_credential() -> None:
    from superlocalmemory.retrieval.remote_reranker import _redact_remote_text

    out = _redact_remote_text(_MEMORY)
    assert _KEY not in out and "[REDACTED:" not in out and "S9t0" not in out


def test_the_hosted_check_transport_screens_what_it_sends() -> None:
    """Defence in depth: the judge redacts before it builds a request, and
    the transport screens again on the final URL."""
    import time

    from superlocalmemory.retrieval.jev_transport import HostedTransport

    bodies: list[bytes] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, json={"ok": True})

    hosted = HostedTransport(transport=httpx.MockTransport(_handler))
    try:
        for kw in ({"json": {"memories": [_MEMORY]}},
                   {"content": json.dumps({"memories": [_MEMORY]}).encode()}):
            _body, status, failure = hosted.post(
                "https://api.typesafe.ai/v1/systemone", {"X": "1"},
                deadline=time.monotonic() + 5, **kw)
            assert (status, failure) == (200, "")
    finally:
        hosted.close()
    assert len(bodies) == 2
    for body in bodies:
        assert _KEY.encode() not in body and b"Hunt3rTwo" not in body


def test_an_incomplete_llm_config_falls_back_to_the_heuristic(wire) -> None:
    """A config object missing fields must not crash the store path."""
    from superlocalmemory.core.summarizer import Summarizer

    incomplete = types.SimpleNamespace(
        mode=types.SimpleNamespace(value="c"),
        llm=types.SimpleNamespace(provider="azure", api_key=_AZURE_KEY, model="gpt-4o",
                                  api_base=_AZURE_BASE, timeout_seconds=30),
    )
    out = Summarizer(incomplete).summarize_cluster(
        [{"content": "Atlas ships on Friday. More text."}])
    assert out == "Atlas ships on Friday." and wire == []
