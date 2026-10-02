# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The dashboard shows the answer check that actually runs.

Under the never-chosen starting mode "auto", the on-device check runs once it
is set up — and the System card said "Off" while it did, and the Settings
radios showed nothing selected (so cancelling a switch threw). These tests
EXECUTE the two renderers' decision functions under node; the full click
paths (confirm, cancel, polling, the background check) were driven in a
headless browser against a stubbed daemon.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "ui" / "js"
_NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(_NODE is None, reason="no JS runtime available")

_HARNESS = r"""
const fs = require('node:fs');
const noop = () => {};
globalThis.window = {addEventListener: noop};
globalThis.document = {addEventListener: noop, getElementById: () => null,
                       querySelector: () => null};
globalThis.fetch = () => Promise.resolve({ok: false, json: () => ({})});
const dash = fs.readFileSync(process.argv[1], 'utf8');
const {answerCheckSummaryText} = new Function(dash + '; return {answerCheckSummaryText};')();
new Function(fs.readFileSync(process.argv[2], 'utf8'))();
const statuses = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(statuses.map(s => ({
  card: answerCheckSummaryText(s),
  radio: window.SLMAnswerCheck.effectiveMode(s),
}))));
"""


def _run(*statuses: dict) -> list[dict]:
    proc = subprocess.run(
        [_NODE, "-e", _HARNESS, str(_UI / "dashboard.js"), str(_UI / "answer-check.js"),
         json.dumps(list(statuses))],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _status(**over) -> dict:
    base = {"mode": "auto", "active": "off", "laya": {"state": "not_installed"},
            "jev": {"provider": "typesafe", "has_key": False,
                    "rerank": {"enabled": False, "active": False, "k": 20}}}
    return {**base, **over}


def test_auto_running_on_this_mac_is_shown_as_on():
    (out,) = _run(_status(active="laya", laya={"state": "ready"}))
    assert out["card"].startswith("On this Mac")
    assert out["radio"] == "laya"


def test_auto_with_nothing_set_up_is_shown_as_off():
    (out,) = _run(_status())
    assert out["card"] is None          # the card then offers "Set up"
    assert out["radio"] == "off"


def test_an_explicit_choice_is_shown_as_chosen_even_when_it_cannot_run():
    (out,) = _run(_status(mode="laya", active="off", laya={"state": "failed"}))
    assert out["card"] == "On this Mac · failed"
    assert out["radio"] == "laya"


def test_the_online_check_and_its_reordering():
    (out,) = _run(_status(mode="jev", active="jev", jev={
        "provider": "openrouter", "has_key": True,
        "rerank": {"enabled": True, "active": True, "k": 20}}))
    assert out["card"] == "Online (OpenRouter) · ready · reordering on"
    assert out["radio"] == "jev"


def test_a_status_without_its_jev_block_does_not_throw():
    (out,) = _run({"mode": "jev", "active": "off", "laya": {"state": "ready"}})
    assert out["card"] == "Online (TypeSafe) · off"
