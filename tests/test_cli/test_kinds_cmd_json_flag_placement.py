# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L3-14: ``--json`` works at either parser level for ``slm kinds``.

argparse writes every action's own default into the namespace while parsing
a subparser, so a nested ``--json`` with the ordinary ``store_true`` default
of False silently overwrote an already-true value from the parent parser --
``slm kinds --json status`` and ``slm kinds status --json`` were documented
as equivalent but only the second actually set ``args.json``. Separately,
``slm kinds backfill --json`` (no action word; ``_backfill`` already treats
a missing one as "status") was an argparse usage error because the
``backfill`` parser itself had no ``--json`` of its own.
"""

from __future__ import annotations

import argparse

import pytest

from superlocalmemory.cli.kinds_cmd import register_kinds_parser


def _parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="slm")
    sub = top.add_subparsers(dest="command")
    register_kinds_parser(sub)
    return top


@pytest.mark.parametrize("argv", [
    ["kinds", "--json", "status"],
    ["kinds", "status", "--json"],
])
def test_json_flag_before_or_after_the_subcommand(argv) -> None:
    args = _parser().parse_args(argv)
    assert args.json is True


def test_json_flag_absent_is_false() -> None:
    args = _parser().parse_args(["kinds", "status"])
    assert args.json is False


@pytest.mark.parametrize("argv", [
    ["kinds", "--json", "backfill", "start"],
    ["kinds", "backfill", "--json", "start"],
    ["kinds", "backfill", "start", "--json"],
])
def test_json_flag_at_any_of_three_levels(argv) -> None:
    args = _parser().parse_args(argv)
    assert args.json is True


def test_backfill_with_json_and_no_action_word_is_accepted() -> None:
    """Hermes' second argv: ``slm kinds backfill --json`` with no action
    word. _backfill() already treats a missing backfill_command as "status";
    before this fix argparse rejected the invocation before _backfill ever
    ran, because the backfill parser had no --json of its own."""
    args = _parser().parse_args(["kinds", "backfill", "--json"])
    assert args.json is True
    assert args.backfill_command is None
