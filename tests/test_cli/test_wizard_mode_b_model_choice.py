# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Mode B offers the Ollama models actually installed, best first (#4.1.22).

Varun: people can run more powerful models locally than the old hardcoded
``llama3.2`` default; the wizard should rank what is actually pulled using
SLM's own extraction test (``core.model_catalog``) instead of a name nobody
would know to type. Never touches a real Ollama server — ``httpx.get`` is
stubbed throughout, mirroring
``tests/test_server/test_picking_an_ollama_model_from_the_dashboard.py``.
"""

from __future__ import annotations

import shutil

import httpx
import pytest

import superlocalmemory.cli.setup_wizard as sw
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def _cfg(tmp_path):
    return SLMConfig.for_mode(Mode.B, base_dir=tmp_path)


def _stub_installed(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(
        200, {"models": [{"name": n} for n in names]},
    ))


def _stub_unreachable(monkeypatch) -> None:
    def _refuse(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", _refuse)


@pytest.fixture(autouse=True)
def _ram_and_ollama_binary(monkeypatch):
    monkeypatch.setattr(sw, "_get_ram_gb", lambda: 24.0)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ollama")


class TestNonInteractiveDefault:
    def test_installed_models_pick_gemma_first(self, tmp_path, monkeypatch) -> None:
        _stub_installed(monkeypatch, ["llama3.2:latest", "gemma3:4b"])
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("non-interactive must never prompt")))

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=False)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.llm.provider == "ollama"
        assert reloaded.llm.model == "gemma3:4b"


class TestInteractiveChoice:
    def test_enter_keeps_the_recommended_default(self, tmp_path, monkeypatch) -> None:
        _stub_installed(monkeypatch, ["llama3.2:latest", "gemma3:4b"])
        prompts = iter(["1", ""])  # server choice, then model choice (Enter)
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: next(prompts))

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=True)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.llm.model == "gemma3:4b"

    def test_typing_2_picks_the_second_listed_model(self, tmp_path, monkeypatch) -> None:
        _stub_installed(monkeypatch, ["llama3.2:latest", "gemma3:4b"])
        prompts = iter(["1", "2"])
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: next(prompts))

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=True)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        # recs[0] is gemma3:4b (best-ranked installed+fitting); recs[1] is
        # llama3.2 (also installed+fitting, worse-ranked) — "2" selects it.
        assert reloaded.llm.model == "llama3.2"

    def test_typing_an_uninstalled_name_saves_it_and_hints_the_pull(
        self, tmp_path, monkeypatch, capsys,
    ) -> None:
        _stub_installed(monkeypatch, ["llama3.2:latest", "gemma3:4b"])
        prompts = iter(["1", "mistral:7b"])
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: next(prompts))

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=True)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.llm.model == "mistral:7b"
        assert "ollama pull mistral:7b" in capsys.readouterr().out


class TestOllamaUnreachable:
    def test_falls_back_to_the_catalogue_default_without_crashing(
        self, tmp_path, monkeypatch,
    ) -> None:
        _stub_unreachable(monkeypatch)
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("non-interactive must never prompt")))

        config = _cfg(tmp_path)
        sw._run_mode_b_server_step(config, interactive=False)

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.llm.provider == "ollama"
        assert reloaded.llm.model == "gemma3:4b"
