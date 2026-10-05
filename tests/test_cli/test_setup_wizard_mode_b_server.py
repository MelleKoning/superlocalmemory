# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Setup wizard's Mode B server step (#112, part 2).

Before this, the interactive wizard only ever wired Mode B to Ollama; a
non-Ollama local server (llama.cpp, vLLM, LM Studio, …) required the
dashboard's Settings pane. ``_run_mode_b_server_step`` now offers both,
mirroring ``test_setup_wizard_laya.py``'s pattern of testing a wizard
sub-step directly rather than driving the whole multi-step ``run_wizard()``.
"""

from __future__ import annotations

import shutil

import superlocalmemory.cli.setup_wizard as sw
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode
from tests.fixtures.stub_llm_server import StubLLMServer


def _cfg(tmp_path):
    config = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    return config


class TestNonInteractiveKeepsOllamaDefault:
    def test_non_interactive_never_prompts_and_saves_ollama(
        self, tmp_path, monkeypatch, capsys,
    ) -> None:
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("non-interactive must never prompt")))
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ollama")

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=False)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.mode is Mode.B
        assert reloaded.llm.provider == "ollama"
        assert "Ollama found" in capsys.readouterr().out


class TestInteractiveDefaultIsStillOllama:
    def test_default_choice_keeps_ollama(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: "1")
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ollama")

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=True)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.mode is Mode.B
        assert reloaded.llm.provider == "ollama"


class TestInteractiveCustomServerChoice:
    def test_choosing_2_delegates_to_custom_endpoint_configuration(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Wiring test: choice "2" must call configure_custom_endpoint_provider
        targeting Mode B, interactively, with no endpoint pre-filled (so IT
        does the prompting)."""
        calls = []

        def _fake_configure(config, **kwargs):
            calls.append((config, kwargs))

        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: "2")
        monkeypatch.setattr(
            "superlocalmemory.cli.provider_custom_endpoint.configure_custom_endpoint_provider",
            _fake_configure,
        )

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=True)

        assert len(calls) == 1
        _config, kwargs = calls[0]
        assert _config is config
        assert kwargs == {
            "endpoint": None, "api_key": None, "model": None,
            "target_mode": "b", "interactive": True,
        }

    def test_choosing_2_end_to_end_saves_mode_b_with_custom_endpoint(
        self, tmp_path, monkeypatch,
    ) -> None:
        """End-to-end: a real stub server, prompts resolved via _prompt, no
        mocking of configure_custom_endpoint_provider itself.

        Two modules read prompts: setup_wizard (the 1/2 server choice) and
        provider_custom_endpoint (endpoint/key/model) — it imported
        ``_prompt``/``is_interactive`` by name, so each must be patched on
        its OWN module; patching only ``sw`` would leave the second
        module's calls hitting the real ``is_interactive()`` (False under
        pytest), silently skipping every prompt.
        """
        import superlocalmemory.cli.provider_custom_endpoint as pce

        with StubLLMServer() as stub:
            endpoint = f"{stub.url}/v1"
            prompts = iter(["2", endpoint, "", "my-local-model"])

            def _next_prompt(*_a, **_k):
                return next(prompts)

            monkeypatch.setattr(sw, "_prompt", _next_prompt)
            monkeypatch.setattr(pce, "_prompt", _next_prompt)
            monkeypatch.setattr(pce, "is_interactive", lambda: True)

            config = _cfg(tmp_path)
            sw._run_mode_b_server_step(config, interactive=True)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.mode is Mode.B
        assert reloaded.llm.provider == "openai"
        assert reloaded.llm.api_base == endpoint
        assert reloaded.llm.api_key == ""
        assert reloaded.llm.model == "my-local-model"
