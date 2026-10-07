# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Letting a person pick a model instead of the wizard picking it for them (#4.1.22).

Varun: people can run more powerful models locally than the old hardcoded
``llama3.2`` default — Ollama users who already pulled something larger or
more faithful should see it offered, ranked by SLM's own extraction test
(``core.model_catalog``), not buried behind a name they'd have to already
know to type. The same is true of Mode C's hosted catalogue: a cheaper or
stronger OpenRouter tier shouldn't require already knowing its id.

Kept out of ``cli/setup_wizard.py`` on purpose: that file is already past the
800-line guideline, so new logic goes in its own small module instead of
growing it further (see ``.backup/4.1.22/LANE-RULES.md``). Both entry points
take their prompt function as a parameter rather than importing
``setup_wizard`` themselves, so existing tests that monkeypatch
``setup_wizard._prompt`` keep working unchanged — the caller passes that same
(possibly patched) callable through.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from superlocalmemory.core import model_catalog

#: ``_prompt(message, default) -> str`` — see ``cli.setup_wizard._prompt``.
PromptFn = Callable[[str, str], str]


def pick_mode_b_model(
    config: Any,
    *,
    interactive: bool,
    prompt: PromptFn,
    installed: list[str] | None,
    ram_gb: float | None,
) -> None:
    """Choose Mode B's local LLM and save it onto ``config.llm``.

    Installed models that fit this machine's memory come first, ranked by
    SLM's extraction test. The catalogue is a recommendation, never a
    restriction: any installed model, or any name at all, can still be typed.
    Non-interactive runs take the top recommendation silently.
    """
    recs = model_catalog.recommend_local_llms(ram_gb or None, installed or [])
    default_model = model_catalog.best_local_llm(ram_gb or None, installed or [])
    installed_ids = {rec.model_id for rec in recs if rec.installed}

    chosen = default_model
    if interactive:
        print()
        print("  Local models (best first):")
        recommended_id = next(
            (rec.model_id for rec in recs if rec.installed and rec.fits), None,
        )
        for n, rec in enumerate(recs, 1):
            tags = []
            if rec.installed:
                tags.append("installed")
            if rec.model_id == recommended_id:
                tags.append("recommended")
            suffix = f" ({', '.join(tags)})" if tags else ""
            print(f"  [{n}] {rec.model_id} — {rec.reason}{suffix}")
        choice = prompt(
            f"  Select a model: number, name, or Enter for {default_model}: ", "",
        ).strip()
        if choice:
            if choice.isdigit() and 1 <= int(choice) <= len(recs):
                chosen = recs[int(choice) - 1].model_id
            else:
                chosen = choice
        if chosen.removesuffix(":latest") not in installed_ids:
            print(f"  Pull it first: ollama pull {chosen}")

    config.llm = dataclasses.replace(config.llm, provider="ollama", model=chosen)


def pick_mode_c_model(
    *, provider_name: str, default_model: str, prompt: PromptFn,
) -> str:
    """Offer the model for a Mode C preset instead of silently taking it.

    OpenRouter lists the hosted catalogue (id, price, advice); the direct
    presets (openai/anthropic/ollama) show the preset model as the default.
    Enter keeps the default; any typed text is used as the model id.
    """
    if provider_name == "openrouter":
        print()
        print("  OpenRouter models:")
        for i, entry in enumerate(model_catalog.HOSTED_LLMS, 1):
            print(f"  [{i}] {entry.id} — {entry.label} — {entry.price} — {entry.advice}")
        choice = prompt(
            f"  Select a model: number, id, or Enter for {default_model}: ", "",
        ).strip()
        if not choice:
            return default_model
        if choice.isdigit() and 1 <= int(choice) <= len(model_catalog.HOSTED_LLMS):
            return model_catalog.HOSTED_LLMS[int(choice) - 1].id
        return choice

    choice = prompt(f"  Model [Enter for {default_model}]: ", "").strip()
    return choice or default_model


__all__ = ["pick_mode_b_model", "pick_mode_c_model"]
