# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The Recall Lab has to show a degraded retrieval path, not just thin results.

The engine now reports what became of each channel and the response carries it.
A status nobody can see would be the same defect wearing a new field, so this
covers the last layer.

Where a JS runtime is available these tests EXECUTE the renderer against a
minimal DOM rather than grepping it, because the interesting properties —
singular vs plural, which statuses read as faults, an unknown status not
crashing the pane — are behaviour, and a static scan cannot see any of them.
The ordering test below stays static: it is a property of the file, and it is
the one thing most likely to be undone by a later edit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_JS = (
    Path(__file__).resolve().parents[2]
    / "src" / "superlocalmemory" / "ui" / "js" / "recall-lab.js"
)

_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="no JS runtime available")

_HARNESS = r"""
const fs = require('node:fs');
const src = fs.readFileSync(process.argv[1], 'utf8');
class El {
  constructor(t){this.tag=t;this.children=[];this.className='';this.style={};this._text='';}
  set textContent(v){this._text=String(v);this.children=[];}
  get textContent(){return this._text + this.children.map(c=>c.textContent).join('');}
  appendChild(c){this.children.push(c);return c;}
}
globalThis.document = {
  createElement: t => new El(t),
  addEventListener(){}, getElementById: () => null,
  createTextNode: t => ({textContent: String(t)}),
};
globalThis.fetch = () => Promise.resolve({json: () => ({})});
const {buildChannelHealth} = new Function(src + '; return {buildChannelHealth};')();
const out = buildChannelHealth(JSON.parse(process.argv[2]));
process.stdout.write(JSON.stringify({
  rendered: out !== null && out !== undefined,
  text: out ? out.textContent : null,
  classes: out ? out.children.map(c => c.className) : null,
}));
"""


def _render(statuses: dict) -> dict:
    proc = subprocess.run(
        [_NODE, "-e", _HARNESS, str(_JS), json.dumps(statuses)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_pane_actually_contains_the_renderer() -> None:
    """"The file exists" is true of an empty file.

    The assertion that matters is that the thing under test is in it, so that
    deleting the renderer fails here rather than in whichever test happens to
    call it next.
    """
    assert _JS.exists(), f"missing UI JS file: {_JS}"
    src = _JS.read_text(encoding="utf-8")
    assert len(src) > 500, f"{_JS} is {len(src)} bytes; that is not the pane"
    assert "buildChannelHealth" in src, (
        "the channel-health renderer is gone from the pane"
    )


@_needs_node
def test_a_failed_channel_is_named_and_called_a_failure() -> None:
    out = _render({"bm25": "error", "temporal": "ok"})
    assert "bm25" in out["text"]
    assert "failed" in out["text"]
    assert "degraded" in out["text"].lower()
    assert any("bg-danger" in c for c in out["classes"])


@_needs_node
def test_a_channel_that_found_nothing_is_not_dressed_up_as_a_failure() -> None:
    """Finding nothing is a legitimate answer and must not read as an outage."""
    out = _render({"bm25": "empty", "temporal": "ok"})
    assert "degraded" not in out["text"].lower()
    assert not any("bg-danger" in c for c in out["classes"])
    assert not any("bg-warning" in c for c in out["classes"])


@_needs_node
def test_an_operators_own_configuration_is_not_a_fault() -> None:
    out = _render({"bm25": "disabled", "hopfield": "not_configured"})
    assert "degraded" not in out["text"].lower()


@_needs_node
def test_a_timeout_is_shown_differently_from_a_crash() -> None:
    timed_out = _render({"bm25": "timeout"})
    crashed = _render({"bm25": "error"})
    assert timed_out["text"] != crashed["text"]
    assert "timed out" in timed_out["text"]


@_needs_node
def test_it_counts_the_broken_channels_in_readable_english() -> None:
    one = _render({"bm25": "error", "temporal": "ok"})
    two = _render({"bm25": "error", "semantic": "no_embedding", "temporal": "ok"})
    assert "1 channel would" in one["text"]
    assert "2 channels would" in two["text"]


@_needs_node
def test_a_status_this_pane_has_never_heard_of_does_not_break_it() -> None:
    """New statuses will be added server-side before this file learns them."""
    out = _render({"future_channel": "some_new_status"})
    assert out["rendered"]
    assert "some_new_status" in out["text"]


@_needs_node
def test_nothing_to_report_renders_nothing() -> None:
    assert _render({})["rendered"] is False


def test_the_strip_renders_before_the_no_results_branch() -> None:
    """Ordering is the load-bearing part and a static property of the file.

    Every channel failing yields exactly zero results, so the answer most in
    need of an explanation is the one that would otherwise show a bare
    'No results found'. Rendering the strip after that early return would make
    the field invisible in precisely the case it was added for.
    """
    src = _JS.read_text(encoding="utf-8")
    strip = src.index("buildChannelHealth(recallLabState.channelStatus)")
    empty_branch = src.index("recallLabState.allResults.length === 0")
    assert strip < empty_branch, (
        "the channel report renders after the no-results early return, so it "
        "is hidden exactly when every channel failed"
    )


def test_the_pane_is_not_an_injection_sink() -> None:
    """Statuses arrive over HTTP; this pane builds DOM, it does not parse HTML."""
    src = _JS.read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in src, f"{sink} in recall-lab.js"
