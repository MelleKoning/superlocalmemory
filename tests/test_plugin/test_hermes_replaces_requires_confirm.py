# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Hermes gates high-impact CLI commands behind a CONFIRM token
(command-inventory.json's ``high_impact`` list) — ``review-correction``
(the undo for a replace) and ``kinds`` (which includes ``kinds set``) both
require it. ``/slm remember ... --replaces <id>`` does the same kind of
destructive, hard-to-undo thing (supersedes a memory in place) but
``remember`` is not in ``high_impact`` — it cannot be, since routine
remembers must stay a single step — so plain ``--replaces`` usage slipped
through with no confirmation step at all (L3-16).
"""

from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE_INIT = REPO / "plugin-src" / "hermes" / "__init__.py"
BUILT_INIT = REPO / "hermes-plugin" / "__init__.py"


def _load_plugin(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("slm_hermes_replaces_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.SlmHermesPlugin(ctx=object())


def _router_cases(path: pathlib.Path):
    plugin = _load_plugin(path)

    # --- --replaces present: must demand CONFIRM, same message shape as
    # the existing high-impact gate. ---
    preview = plugin.slash_router('remember "new fact" --replaces fact-123')
    assert preview.startswith("Preview required."), preview
    assert "CONFIRM" in preview

    # --- CONFIRM present alongside --replaces: gate passes (it will go on
    # to look for a real `slm` binary and fail differently — that failure,
    # not a second "Preview required", is what proves the gate opened). ---
    confirmed = plugin.slash_router('remember "new fact" --replaces fact-123 CONFIRM')
    assert not confirmed.startswith("Preview required."), confirmed

    # --- plain remember (no --replaces): must NOT be gated — remember is
    # not in high_impact and must stay a single step for routine writes. ---
    plain = plugin.slash_router('remember "just a fact"')
    assert not plain.startswith("Preview required."), plain


def test_replaces_requires_confirm_in_source() -> None:
    _router_cases(SOURCE_INIT)


def test_replaces_requires_confirm_in_built_plugin() -> None:
    _router_cases(BUILT_INIT)
