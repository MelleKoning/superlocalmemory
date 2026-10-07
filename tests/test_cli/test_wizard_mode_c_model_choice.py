# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Mode C (``configure_provider``) offers the model instead of taking the
preset silently (#4.1.22).

OpenRouter users see SLM's hosted catalogue (``core.model_catalog.HOSTED_LLMS``)
so a cheaper or stronger tier doesn't require already knowing its id; Enter
keeps the preset default, matching the wizard's existing zero-prompt-by-default
contract for anyone who just wants to move on.
"""

from __future__ import annotations

import superlocalmemory.cli.setup_wizard as sw
from superlocalmemory.core import model_catalog
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


def _cfg(tmp_path):
    return SLMConfig.for_mode(Mode.A, base_dir=tmp_path)


def _run(tmp_path, monkeypatch, prompts: list[str]):
    monkeypatch.setattr(sw, "is_interactive", lambda: True)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    answers = iter(prompts)
    monkeypatch.setattr(sw, "_prompt", lambda *a, **k: next(answers))

    config = _cfg(tmp_path)
    sw.configure_provider(config)
    return SLMConfig.load(tmp_path / "config.json")


class TestOpenRouterModelChoice:
    def test_enter_keeps_the_preset_model(self, tmp_path, monkeypatch) -> None:
        # [4] openrouter, "" API key, "" model (Enter)
        reloaded = _run(tmp_path, monkeypatch, ["4", "", ""])

        assert reloaded.llm.provider == "openrouter"
        assert reloaded.llm.model == model_catalog.DEFAULT_HOSTED_LLM

    def test_typed_id_is_saved(self, tmp_path, monkeypatch) -> None:
        typed = "anthropic/claude-sonnet-5.5"
        reloaded = _run(tmp_path, monkeypatch, ["4", "", typed])

        assert reloaded.llm.provider == "openrouter"
        assert reloaded.llm.model == typed
