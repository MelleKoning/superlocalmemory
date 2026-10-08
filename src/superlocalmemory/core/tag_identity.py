# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What a tag label names - the one rule every tag comparison uses.

4.1.22: tags live exactly where 4.1.21 left them,
``memories.metadata_json -> '$.tags'`` - no schema migration ships with this
(a new ``memory.db`` migration forces a full safety copy, ~2 GB on a real
store, at upgrade). What changes is that a label is now matched by its
CANONICAL identity, not by a raw string comparison, and three stored shapes
are parsed the same way everywhere instead of each caller guessing at its own.

Stored shapes, observed on the real M5 store:

* most rows: a comma-separated string (``"project-x,db"``), the shape every
  write path (HTTP ``/remember``'s ``tags: str`` field, the MCP ``remember``
  tool, the CLI's ``--tags``) has always produced;
* a few rows: a string that is itself a JSON array (``'["a", "b"]'``) - SQLite
  has no array type, so a value originally written as a real list comes back
  from ``json_extract`` as this string;
* a caller (MCP, in-process) may also pass a real Python list/tuple directly,
  never round-tripped through JSON text at all.

Canonical identity (``tag_key``): Unicode NFC, trimmed, internal whitespace
collapsed to one space, casefolded. Punctuation is KEPT - "v4.1.21",
"token-optimization" and "#149" are meaningful labels, not noise to strip.
This mirrors ``core.project_identity.project_key`` in spirit (one small
function every tag-aware module calls, so two of them can never disagree)
but keeps punctuation, where a project path's separators are structural and
a tag's punctuation is part of the label itself.

Parsing never invents a tag from free text - it only ever splits what was
already stored or passed in, never, for example, pulls a hashtag out of a
memory's content.
"""

from __future__ import annotations

import json
import re
import unicodedata

#: Longest label kept, before normalisation. A label longer than any real one
#: is not a tag; bounds the cost of normalising a hostile input.
MAX_TAG_CHARS = 200

_WS = re.compile(r"\s+")


def tag_key(value: object) -> str | None:
    """The identity of the tag label ``value`` names, or None when it names
    none.

    "Token-Optimization", " token-optimization ", "token  optimization" (two
    spaces) and the NFD-decomposed form of an accented label all give the
    same key. Empty or whitespace-only text, and anything that is not a
    string, names no tag.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()[:MAX_TAG_CHARS]
    if not text:
        return None
    text = _WS.sub(" ", text)
    text = unicodedata.normalize("NFC", text).casefold()
    return text or None


def parse_tag_values(raw: object) -> list[str]:
    """The tag labels ``raw`` names, in encounter order, before de-duplication.

    ``raw`` may be:

    * a real list/tuple/set of labels - an MCP caller may pass a JSON array
      directly, never round-tripped through text;
    * a string that, once stripped, starts with ``"["`` and parses as a JSON
      list - a value an earlier save stored that way;
    * otherwise, a comma-separated string - the common stored shape, and the
      plain-text form an HTTP/CLI caller passes. A label that itself contains
      a comma cannot be expressed this way; pass a list instead (every
      surface documents this).

    Each item is cleaned (NFC-normalised, trimmed, length-bounded) but kept
    in its ORIGINAL casing - this is the display form; call :func:`tag_key`
    on each item to compare or de-duplicate. Never raises: an item that is
    not a string, or that cleans to empty, is silently dropped, and ``raw``
    being ``None`` or an unrecognised type gives ``[]``. Never invents a
    label from free text - only ever splits what was given.
    """
    if raw is None:
        return []
    items: list[object]
    if isinstance(raw, (list, tuple, set, frozenset)):
        items = list(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        items = None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                items = parsed
        if items is None:
            items = text.split(",")
    else:
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = unicodedata.normalize("NFC", item.strip()[:MAX_TAG_CHARS])
        if cleaned:
            out.append(cleaned)
    return out


def tag_keys(raw: object) -> list[str]:
    """The de-duplicated canonical keys ``raw`` names, in first-seen order.

    Composes :func:`parse_tag_values` and :func:`tag_key`; every caller that
    only needs identity (matching, membership) should call this rather than
    re-implementing the parse-then-key pipeline.
    """
    seen: dict[str, None] = {}
    for item in parse_tag_values(raw):
        key = tag_key(item)
        if key is not None:
            seen.setdefault(key, None)
    return list(seen.keys())


__all__ = ["MAX_TAG_CHARS", "parse_tag_values", "tag_key", "tag_keys"]
