# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Typed values in memory text: measurements are not dates.

A fuzzy date parser will read almost any number as a calendar date: ``4.1.21``
becomes 1 April 2021, ``port 8765`` the year 8765, ``$12.50`` the 12th of this
month, ``1,234`` the year 234. A memory that said "recalled in 2004.6 ms" ended
up stored as a June 2004 date. This module decides, mechanically, which strings
may be handed to a date parser at all, and lists the typed numbers a text
carries so a derived fact can be checked against its source.

Pure functions, no I/O, no model.
"""

from __future__ import annotations

import re

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_WEEKDAY = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?"
_CAL_UNIT = r"(?:days?|weeks?|months?|years?|fortnight|quarter)"
_SEASON = r"(?:spring|summer|autumn|fall|winter)"

_MONTH_RE = re.compile(rf"\b{_MONTH}\b\.?", re.IGNORECASE)
_WEEKDAY_RE = re.compile(rf"\b{_WEEKDAY}\b", re.IGNORECASE)
_RELATIVE_RE = re.compile(
    rf"\b(?:yesterday|today|tomorrow|tonight|"
    rf"(?:last|next|this|coming|past)\s+(?:{_WEEKDAY}|{_CAL_UNIT}|{_SEASON}|weekend|morning|"
    rf"afternoon|evening|night)|"
    rf"\d+\s+{_CAL_UNIT}\s+(?:ago|later|from\s+now)|in\s+\d+\s+{_CAL_UNIT}|"
    rf"the\s+other\s+day|the\s+day\s+before(?:\s+yesterday)?|a\s+while\s+ago|recently)\b",
    re.IGNORECASE,
)
#: Numeric calendar shapes. Dotted forms need a four-digit year at one end, so a
#: version such as 4.1.21 never qualifies.
_NUMERIC_DATE_RE = re.compile(
    r"(?<![\d.])(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{4}/\d{1,2}/\d{1,2}"
    r"|\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})"
    r"|\d{1,2}\.\d{1,2}\.\d{4}"
    r"|\d{4}\.\d{1,2}\.\d{1,2}"
    r")(?![\d.]*\d)"
)

#: Units that may follow a number after a space ("2004.6 ms", "3 GB").
_UNIT_WORD = (
    r"(?:ms|msec|millis(?:econds?)?|milliseconds?|µs|us|ns|sec|secs|seconds?|"
    r"min|mins|minutes?|hr|hrs|hours?|"
    r"kb|mb|gb|tb|kib|mib|gib|tib|bytes?|tokens?|tok/s|rps|qps|req/s|fps|"
    r"px|pt|em|rem|°c|°f|kg|mg|km|cm|mm|"
    r"ghz|mhz|hz|watts?|kw|ma|mah|dpi|ppm|bps|kbps|mbps|gbps)"
)
#: One-letter units count only when attached ("3s", "10k", "2x"), so a hex id
#: or "rank 1 m..." is never read as a measurement.
_UNIT_LETTER = r"(?:s|m|h|k|x|b|c|f|g|w|v)"
_CURRENCY = r"(?:[$€£¥₹]|usd|eur|gbp|inr|rs\.?)"
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"

#: Typed number tokens, longest shapes first. Each match is one typed value.
_TYPED_NUMBER_RE = re.compile(
    r"(?<![\w.#])(?:"
    rf"v\d+(?:\.\d+)*"                                     # v2, v2.3, v4.1.21
    r"|\d+\.\d+\.\d+(?:\.\d+)*"                            # 4.1.21
    rf"|{_CURRENCY}\s?(?:{_NUM})"                           # $12.50, ₹1,200
    rf"|(?:{_NUM})\s?%"                                     # 45%, 3.5 %
    rf"|(?:{_NUM})\s?{_UNIT_WORD}(?!\w)"                   # 2004.6 ms, 3.0 sec
    rf"|(?:{_NUM}){_UNIT_LETTER}(?!\w)"                     # 3s, 10k, 2x
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"                       # 1,234
    r"|\d+\.\d+"                                           # 2004.6
    r")"
    r"|#\d+"                                               # issue #149
    r"|\bport\s+\d+",                                      # port 8765
    re.IGNORECASE,
)

#: A whole string that is one typed measurement and nothing else.
_WHOLE_MEASUREMENT_RE = re.compile(
    rf"^\s*(?:"
    rf"v?\d+(?:\.\d+){{2,}}|v\d+(?:\.\d+)?"
    rf"|{_CURRENCY}\s?(?:{_NUM})"
    rf"|(?:{_NUM})\s?%"
    rf"|(?:{_NUM})\s?{_UNIT_WORD}|(?:{_NUM}){_UNIT_LETTER}|(?:{_NUM})\s+s"
    rf"|\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?"
    rf"|\d+(?:\.\d+)?"
    rf"|#\d+|port\s+\d+"
    rf")\s*$",
    re.IGNORECASE,
)

#: ISO calendar tokens a derived fact may carry (YYYY-MM-DD or YYYY-MM).
ISO_DATE_TOKEN_RE = re.compile(r"(?<![\d.\-])(\d{4})-(\d{2})(?:-(\d{2}))?(?![\d\-])")


def has_relative_time(text: str) -> bool:
    """True when ``text`` states a relative time ("yesterday", "next Tuesday")."""
    return bool(text) and bool(_RELATIVE_RE.search(text))


def is_typed_measurement(raw: str | None) -> bool:
    """True when ``raw`` is one typed number (duration, version, money, id...)."""
    if not raw:
        return False
    if _NUMERIC_DATE_RE.fullmatch(raw.strip()):
        return False  # 06.10.2026 is a date, not a version
    return bool(_WHOLE_MEASUREMENT_RE.match(raw))


def looks_like_calendar_date(raw: str | None) -> bool:
    """True when ``raw`` names a calendar date, a month, a weekday or a relative time.

    This is the only kind of string a date parser may see. A bare number, a
    decimal, a version or a measurement never qualifies.
    """
    if not raw or not raw.strip():
        return False
    if is_typed_measurement(raw):
        return False
    return bool(
        _NUMERIC_DATE_RE.search(raw)
        or _MONTH_RE.search(raw)
        or _WEEKDAY_RE.search(raw)
        or _RELATIVE_RE.search(raw)
    )


def _normalize_number(token: str) -> str:
    """Canonical form for comparing typed values across texts."""
    value = re.sub(r"\s+", "", token.lower())
    value = value.replace(",", "")
    if value.startswith("port"):
        value = "port" + value[4:]
    return value


def typed_number_tokens(text: str) -> frozenset[str]:
    """Every typed number in ``text``, normalized ("1,234" -> "1234", "2004.6 ms" -> "2004.6ms")."""
    if not text:
        return frozenset()
    return frozenset(_normalize_number(m.group(0)) for m in _TYPED_NUMBER_RE.finditer(text))


_CORE_RE = re.compile(r"\d[\d,]*(?:\.\d+)*")


def _canonical_core(core: str) -> str:
    """"1,234" -> "1234"; "3.0" -> "3"; "12.50" -> "12.5"; versions are kept as written."""
    value = core.replace(",", "")
    if value.count(".") == 1:
        value = value.rstrip("0").rstrip(".") if "." in value else value
    return value


def typed_number_cores(text: str) -> frozenset[str]:
    """The numeric value inside every typed number ("2004.6 ms" -> "2004.6", "$12.50" -> "12.5")."""
    cores: set[str] = set()
    if not text:
        return frozenset()
    for match in _TYPED_NUMBER_RE.finditer(text):
        found = _CORE_RE.search(match.group(0))
        if found:
            cores.add(_canonical_core(found.group(0)))
    return frozenset(cores)


def bare_numbers(text: str) -> frozenset[str]:
    """Every number in ``text`` in canonical form (support evidence only)."""
    if not text:
        return frozenset()
    return frozenset(_canonical_core(m.group(0)) for m in _CORE_RE.finditer(text))


def iso_date_tokens(text: str) -> frozenset[str]:
    """ISO calendar tokens (YYYY-MM-DD / YYYY-MM) that appear in ``text``."""
    if not text:
        return frozenset()
    return frozenset(m.group(0) for m in ISO_DATE_TOKEN_RE.finditer(text))


__all__ = [
    "ISO_DATE_TOKEN_RE",
    "bare_numbers",
    "has_relative_time",
    "iso_date_tokens",
    "is_typed_measurement",
    "looks_like_calendar_date",
    "typed_number_cores",
    "typed_number_tokens",
]
