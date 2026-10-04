# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every ``#anchor`` link in README.md and docs/ must land on a real heading.

Audit 4.1.20 L3 found four that did not: two pointed at README sections that a
rewrite renamed, one at a README "Quick Start" that no longer exists, and one at
``{#remote}`` -- heading-attribute syntax GitHub does not support, so the
"anchor" was literal text in the heading and the link from
distributed-deployment.md went nowhere. A reader clicking any of them lands at
the top of a long page with no idea where to look.

Slugs follow GitHub's rules: lower-case, punctuation other than ``-`` and
``_`` dropped, each space becomes ``-``, and a repeated heading gets ``-1``,
``-2``... Explicit ``<a id="...">`` / ``<a name="...">`` targets count too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LINK = re.compile(r"\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HTML_ID = re.compile(r"<a\s+[^>]*\b(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", re.I)


def _strip_inline(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)       # links -> label
    text = re.sub(r"<[^>]+>", "", text)                          # html tags
    return text.replace("`", "")


def github_slug(heading: str) -> str:
    text = _strip_inline(heading).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def _lines_outside_fences(text: str):
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield line


def anchors_of(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    anchors: set[str] = set(_HTML_ID.findall(text))
    for line in _lines_outside_fences(text):
        m = _HEADING.match(line)
        if not m:
            continue
        slug = github_slug(m.group(2))
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return anchors


def anchor_links(path: Path):
    """(line_no, target_file, fragment) for every link carrying a fragment."""
    text = path.read_text(encoding="utf-8")
    fenced = False
    for no, line in enumerate(text.splitlines(), 1):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        for target in _LINK.findall(re.sub(r"`[^`]*`", "", line)):
            if "#" not in target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
                continue  # no fragment, or an external URL
            file_part, fragment = target.split("#", 1)
            dest = path if not file_part else (path.parent / file_part).resolve()
            yield no, dest, fragment


def _broken() -> list[str]:
    cache: dict[Path, set[str]] = {}
    problems = []
    for src in SOURCES:
        for no, dest, fragment in anchor_links(src):
            where = f"{src.relative_to(ROOT)}:{no}"
            if dest.suffix.lower() != ".md":
                continue
            if not dest.exists():
                problems.append(f"{where} -> {dest.name} does not exist")
                continue
            if dest not in cache:
                cache[dest] = anchors_of(dest)
            if fragment.lower() not in cache[dest]:
                problems.append(f"{where} -> {dest.relative_to(ROOT)}#{fragment}")
    return problems


def test_every_doc_anchor_resolves() -> None:
    assert _broken() == []


def test_no_heading_uses_unsupported_attribute_syntax() -> None:
    """``## Title {#id}`` renders the braces literally on GitHub."""
    offenders = []
    for src in SOURCES:
        for line in _lines_outside_fences(src.read_text(encoding="utf-8")):
            if _HEADING.match(line) and re.search(r"\{#[^}]+\}\s*$", line):
                offenders.append(f"{src.relative_to(ROOT)}: {line.strip()}")
    assert offenders == []


@pytest.mark.parametrize("heading, slug", [
    ("MCP memory server: tool profiles", "mcp-memory-server-tool-profiles"),
    ("Answer Check: memory that says \"I don't have that\"",
     "answer-check-memory-that-says-i-dont-have-that"),
    ("Benchmarks (V3)", "benchmarks-v3"),
    ("Try Bounded Loops (v3.8.0)", "try-bounded-loops-v380"),
    ("`slm list` and `--json`", "slm-list-and---json"),
])
def test_slugs_match_github(heading: str, slug: str) -> None:
    assert github_slug(heading) == slug
