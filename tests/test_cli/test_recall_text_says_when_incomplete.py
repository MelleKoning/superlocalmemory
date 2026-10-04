# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm recall`` text must not print a confident "no match" for a search that
did not look everywhere (4.1.20 WP10)."""

from __future__ import annotations

from argparse import Namespace

from superlocalmemory.cli.recall_text import empty_result_line, incomplete_line


def _warming() -> dict:
    return {
        "results": [], "no_confident_match": True,
        "incomplete_channels": ["hopfield", "semantic", "spreading_activation"],
        "channel_status": {"semantic": "warming", "hopfield": "warming",
                           "spreading_activation": "warming", "bm25": "empty"},
    }


def test_a_warming_recall_is_never_a_confident_no_match() -> None:
    result = _warming()
    assert empty_result_line(result) != "No confident match."
    line = incomplete_line(result)
    assert "still loading" in line
    assert "semantic" in line


def test_a_complete_recall_reads_exactly_as_before() -> None:
    assert incomplete_line({"incomplete_channels": []}) == ""
    assert empty_result_line({"no_confident_match": True}) == "No confident match."
    assert empty_result_line({}) == "No matching memories found."


def test_a_timed_out_channel_names_the_other_reason() -> None:
    line = incomplete_line({"incomplete_channels": ["temporal"],
                            "channel_status": {"temporal": "timeout"}})
    assert "did not finish" in line and "temporal" in line


def _run_cli(monkeypatch, capsys, result: dict) -> str:
    from superlocalmemory.cli import commands, daemon

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", lambda *a, **k: result)
    commands.cmd_recall(Namespace(query="When is it?", limit=5, json=False))
    return capsys.readouterr().out


def test_the_cli_prints_the_note_with_no_results(monkeypatch, capsys) -> None:
    out = _run_cli(monkeypatch, capsys, _warming())
    assert "No confident match." not in out
    assert "Incomplete search" in out


def test_the_cli_prints_the_note_under_results(monkeypatch, capsys) -> None:
    result = {**_warming(), "results": [{"content": "Halcyon is 14 March", "score": 0.5}]}
    out = _run_cli(monkeypatch, capsys, result)
    assert "Halcyon is 14 March" in out
    assert "Incomplete search" in out
