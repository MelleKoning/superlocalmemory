# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A one-shot command that never recalls builds no answer check.

With the daemon down, ``slm list`` built a full engine — and with it the
on-device check, which started loading a ~1 GB model it would never use. The
check is built idle now in any case; commands that never recall do not build it
at all (no install detection, no interpreter probe). Commands that DO recall
keep it.
"""

from __future__ import annotations

import ast
import contextlib
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.cli import commands

#: Commands that build a full engine and never recall.
NEVER_RECALL = {"_cmd_db_reembed", "cmd_list", "cmd_health", "cmd_observe", "cmd_decay",
                "cmd_quantize", "cmd_consolidate", "cmd_soft_prompts"}
#: Commands that build one to recall with: the check stays as configured.
RECALL = {"cmd_trace", "cmd_session_context"}
#: A LIGHT engine has no retrieval engine, so no check to build.
LIGHT_ONLY = {"cmd_status"}


def _engine_constructions() -> dict[str, list[ast.Call]]:
    tree = ast.parse(Path(commands.__file__).read_text(encoding="utf-8"))
    found: dict[str, list[ast.Call]] = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "MemoryEngine"):
                found.setdefault(fn.name, []).append(node)
    return found


def _wrapped(call: ast.Call) -> bool:
    return bool(call.args) and isinstance(call.args[0], ast.Call) and \
        getattr(call.args[0].func, "id", "") == "_without_answer_check"


def test_every_engine_construction_is_classified() -> None:
    assert set(_engine_constructions()) == NEVER_RECALL | RECALL | LIGHT_ONLY, \
        "a new command builds an engine: decide whether it recalls"


@pytest.mark.parametrize("name", sorted(NEVER_RECALL))
def test_a_command_that_never_recalls_builds_no_check(name) -> None:
    assert all(_wrapped(c) for c in _engine_constructions()[name])


@pytest.mark.parametrize("name", sorted(RECALL))
def test_a_command_that_recalls_keeps_its_check(name) -> None:
    assert not any(_wrapped(c) for c in _engine_constructions()[name])


def test_slm_list_builds_its_engine_with_the_check_off(monkeypatch, capsys) -> None:
    from superlocalmemory.core import engine as engine_mod

    seen: list = []

    class _Stop(RuntimeError):
        pass

    def capture(config, *a, **k):
        seen.append(config.retrieval.sufficiency_judge)
        raise _Stop("enough")

    monkeypatch.setattr(engine_mod, "MemoryEngine", capture)
    with contextlib.suppress(SystemExit):  # the command reports the failure and exits
        commands.cmd_list(Namespace(json=True, limit=1))
    assert seen == ["off"]
