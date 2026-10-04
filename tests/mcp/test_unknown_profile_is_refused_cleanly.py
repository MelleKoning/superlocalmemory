# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""An unknown MCP profile name is refused with the valid names, never a traceback.

``SLM_MCP_PROFILE=all`` crashed ``slm mcp`` with a ValueError traceback from
module import. ``all`` was never a documented name (every tool is ``whole``),
and unknown names stay refused so a typo cannot widen the tool surface, but
the refusal is now one line on stderr with exit code 2. ``slm connect
--profile`` checks the same names before writing anything into an IDE config.
"""

from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_VALID = ("core", "code", "full", "power", "mesh", "whole")


def test_slm_mcp_refuses_all_with_the_valid_names(monkeypatch, capsys) -> None:
    from superlocalmemory.cli.commands import cmd_mcp

    monkeypatch.setenv("SLM_MCP_PROFILE", "all")
    # A stand-in server module: nothing may start a real stdio server here.
    fake_module = types.SimpleNamespace(server=MagicMock())
    monkeypatch.setitem(sys.modules, "superlocalmemory.mcp.server", fake_module)
    # Never let a test reach the machine-wide orphan reaper.
    with (
        patch("superlocalmemory.infra.process_reaper.find_orphans", return_value=[]),
        pytest.raises(SystemExit) as exited,
    ):
        cmd_mcp(Namespace())

    assert exited.value.code == 2
    fake_module.server.run.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout is the JSON-RPC channel
    assert "'all'" in captured.err
    assert all(name in captured.err for name in _VALID)
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("name", ["", "whole", "WHOLE", "core", " full ", "code34", "whole94"])
def test_documented_names_and_aliases_are_accepted(name: str) -> None:
    from superlocalmemory.mcp.profile_names import canonical_profile

    assert canonical_profile(name) in ("", *_VALID)


def test_the_resolver_raises_the_named_error() -> None:
    from superlocalmemory.mcp.profile_names import UnknownProfileError
    from superlocalmemory.mcp.server import _PROFILE_DEFINITIONS, _resolve_profile_allowed

    with pytest.raises(UnknownProfileError, match=r"'all'.*core.*whole"):
        _resolve_profile_allowed("all", _PROFILE_DEFINITIONS, frozenset())


def test_connect_refuses_an_unknown_profile_before_writing(tmp_path: Path) -> None:
    from superlocalmemory.hooks.portable_kit import connect_ide

    result = connect_ide("cursor", home=tmp_path, profile="all")

    assert result["error"] and "'all'" in result["error"]
    assert all(name in result["error"] for name in _VALID)
    assert not any(tmp_path.rglob("*.json"))


def test_connect_still_accepts_whole(tmp_path: Path) -> None:
    from superlocalmemory.hooks.portable_kit import connect_ide

    result = connect_ide("cursor", home=tmp_path, profile="whole")

    assert result["error"] is None
