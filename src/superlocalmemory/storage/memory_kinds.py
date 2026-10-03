# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The nine-kind memory vocabulary — the only place a kind string lives.

LLD §1 (memory-kinds-4.1.19) names this module the frozen cross-module
contract: every other module that needs a kind imports it from here rather
than spelling one of the nine values itself. That is deliberate, not tidy —
``MemoryKind`` is a closed set with an authority order (``AUTHORITY``) and a
confirmed/suggested split (``CONFIRMED_SOURCES``) that several writers must
agree on byte-for-byte. A second copy of the string ``"decision"`` living in
another file is exactly the drift this module exists to prevent.

Nothing here touches a database connection, a judge, or a worker. This is a
pure data-and-rules module so it can be imported from storage, encoding, core,
server and the dashboard-facing CLI without pulling any of those in.

Invariants this module exists to uphold (full list: LLD §1):

- I1 — a write path never fails because of a kind. Every function here that
  takes untrusted input (``parse_kind``, ``is_confirmed``, ``kind_fields``)
  is total: it returns a sentinel (``None`` / ``"untyped"``) rather than
  raising, for any input including ``None``, numbers, empty strings and
  adversarial text.
- I2 — ``COARSE`` is the single mapping from a confirmed kind to the legacy
  ``fact_type`` value every pre-4.1.19 reader still understands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class MemoryKind(str, Enum):
    """The nine kinds a memory can be tagged with (LLD §2)."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    STATUS = "status"
    OPINION = "opinion"
    RULE = "rule"
    DECISION = "decision"
    PROCEDURE = "procedure"
    PROSPECTIVE = "prospective"
    CORRECTION = "correction"


class KindSource(str, Enum):
    """Who assigned a kind, and therefore how much it is trusted (AUTHORITY)."""

    USER = "user"
    CALLER = "caller"
    MODEL_LLM = "model:llm"
    MODEL_LAYA = "model:laya"
    MODEL_JEV = "model:jev"
    RULES = "rules"
    LEGACY = "legacy"


#: Sources whose kind is trusted enough to drive behaviour (LLD I6) and to be
#: shown as settled rather than "suggested". Everything else is a suggestion
#: that changes no behaviour until it is confirmed.
CONFIRMED_SOURCES: frozenset[KindSource] = frozenset({KindSource.USER, KindSource.CALLER})

_CONFIRMED_VALUES: frozenset[str] = frozenset(s.value for s in CONFIRMED_SOURCES)

#: Higher wins. Used by the upsert CASE predicate (LLD §4.4) and by any code
#: comparing two assignments for the same fact. Two sources never tie except
#: the three model backends, which are deliberately equal — none of them is
#: trusted over another, only over `rules`/`legacy` and never over a human.
AUTHORITY: dict[KindSource, int] = {
    KindSource.USER: 6,
    KindSource.CALLER: 5,
    KindSource.MODEL_LLM: 3,
    KindSource.MODEL_JEV: 3,
    KindSource.MODEL_LAYA: 3,
    KindSource.RULES: 2,
    KindSource.LEGACY: 1,
}

#: Total: every ``MemoryKind`` member maps to one of the four pre-4.1.19
#: ``FactType`` values, so a confirmed row always has a legible `fact_type`
#: for every reader that has never heard of kinds (LLD I2).
COARSE: dict[MemoryKind, str] = {
    MemoryKind.SEMANTIC: "semantic",
    MemoryKind.EPISODIC: "episodic",
    MemoryKind.STATUS: "episodic",
    MemoryKind.OPINION: "opinion",
    MemoryKind.RULE: "semantic",
    MemoryKind.DECISION: "episodic",
    MemoryKind.PROCEDURE: "semantic",
    MemoryKind.PROSPECTIVE: "prospective",
    MemoryKind.CORRECTION: "semantic",
}

#: Pre-4.1.19 ``fact_type`` spellings (including two retired names, M046/M048)
#: mapped onto the nearest kind. Used only for *display* of a row nothing has
#: ever classified (`kind_fields` legacy fallback) and by backfill's own
#: "legacy" bookkeeping — never written into `memory_kind` as a confirmed
#: value.
LEGACY_TO_KIND: dict[str, MemoryKind] = {
    "semantic": MemoryKind.SEMANTIC,
    "episodic": MemoryKind.EPISODIC,
    "opinion": MemoryKind.OPINION,
    "prospective": MemoryKind.PROSPECTIVE,
    "temporal": MemoryKind.PROSPECTIVE,   # M046 legacy spelling
    "world": MemoryKind.SEMANTIC,         # v2_migrator.py:36
    "experience": MemoryKind.EPISODIC,    # v2_migrator.py:36
}

#: Human-facing label for dashboards, CLI tables and MCP responses (LLD §2).
LABELS: dict[MemoryKind, str] = {
    MemoryKind.SEMANTIC: "Fact",
    MemoryKind.EPISODIC: "Event",
    MemoryKind.STATUS: "Current state",
    MemoryKind.OPINION: "Preference or view",
    MemoryKind.RULE: "Standing rule",
    MemoryKind.DECISION: "Decision",
    MemoryKind.PROCEDURE: "How-to",
    MemoryKind.PROSPECTIVE: "Plan or to-do",
    MemoryKind.CORRECTION: "Correction",
}

#: Free-form synonyms a caller might type, normalised to one of the nine
#: canonical values (LLD §2's "Accepted aliases" column). Keys are already
#: normalised (lower-case, hyphens not underscores) — ``parse_kind`` performs
#: the same normalisation on its input before looking a value up here.
ALIASES: dict[str, MemoryKind] = {
    "fact": MemoryKind.SEMANTIC,
    "knowledge": MemoryKind.SEMANTIC,
    "event": MemoryKind.EPISODIC,
    "experience": MemoryKind.EPISODIC,
    "state": MemoryKind.STATUS,
    "project-state": MemoryKind.STATUS,
    "progress": MemoryKind.STATUS,
    "preference": MemoryKind.OPINION,
    "view": MemoryKind.OPINION,
    "instruction": MemoryKind.RULE,
    "constraint": MemoryKind.RULE,
    "policy": MemoryKind.RULE,
    "convention": MemoryKind.RULE,
    "choice": MemoryKind.DECISION,
    "how-to": MemoryKind.PROCEDURE,
    "howto": MemoryKind.PROCEDURE,
    "steps": MemoryKind.PROCEDURE,
    "recipe": MemoryKind.PROCEDURE,
    "plan": MemoryKind.PROSPECTIVE,
    "todo": MemoryKind.PROSPECTIVE,
    "to-do": MemoryKind.PROSPECTIVE,
    "commitment": MemoryKind.PROSPECTIVE,
    "task": MemoryKind.PROSPECTIVE,
    "erratum": MemoryKind.CORRECTION,
    "fix": MemoryKind.CORRECTION,
}

#: Carried inside ``RememberRequest.metadata`` so a caller-declared kind rides
#: the existing admission journal without a schema or journal-format change.
METADATA_KEY = "_slm_memory_kind"

#: The five nullable columns M052 (WP-1) adds to ``atomic_facts``. The only
#: place their names are spelled besides the migration's own DDL and the SQL
#: this contract explicitly allows (LLD §4.4, §7.3, §4.5).
KIND_COLUMNS: tuple[str, ...] = (
    "memory_kind",
    "memory_kind_source",
    "memory_kind_confidence",
    "memory_kind_recipe",
    "memory_kind_at",
)


def parse_kind(value: object) -> MemoryKind | None:
    """Best-effort parse of a caller-, model- or storage-supplied kind value.

    Never raises. Accepts a canonical value, a known alias, extra
    whitespace, mixed case, and underscores where a hyphenated alias expects
    a hyphen (``project_state`` reads the same as ``project-state``). Any
    other input — ``None``, a number, an empty string, an unrecognised word,
    adversarial text — returns ``None`` rather than raising, because a bad
    kind must never be the reason a write fails (I1).
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        return None
    try:
        return MemoryKind(normalized)
    except ValueError:
        pass
    return ALIASES.get(normalized)


def is_confirmed(source: str | None) -> bool:
    """Whether a stored ``memory_kind_source`` value counts as confirmed.

    Never raises on a non-string, a value from a future source this build
    does not know about, or ``None`` (an untyped row) — all answer ``False``.
    """
    if not isinstance(source, str):
        return False
    return source in _CONFIRMED_VALUES


@dataclass(frozen=True, slots=True)
class KindAssignment:
    """One kind decision for one fact: what, who said so, how sure, by what recipe."""

    kind: MemoryKind
    source: KindSource
    confidence: float | None
    recipe: str

    def as_columns(self, now_iso: str) -> dict[str, object]:
        """The five ``KIND_COLUMNS`` values this assignment writes."""
        return {
            "memory_kind": self.kind.value,
            "memory_kind_source": self.source.value,
            "memory_kind_confidence": self.confidence,
            "memory_kind_recipe": self.recipe,
            "memory_kind_at": now_iso,
        }


def _read(obj: object, name: str) -> object:
    """Attribute or mapping-key read, tolerant of either shape. Never raises."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _value_of(value: object) -> object:
    """Unwrap a stored enum back to its plain value; everything else passes through."""
    if isinstance(value, Enum):
        return value.value
    return value


def kind_fields(obj: object, *, display_min_confidence: float = 0.20) -> dict[str, object]:
    """THE serializer every surface uses: MCP, CLI JSON, HTTP and the dashboard.

    Reads ``memory_kind``, ``memory_kind_source``, ``memory_kind_confidence``
    and ``fact_type`` off ``obj`` (attribute or mapping key — works on an
    ``AtomicFact``, a ``sqlite3.Row`` wrapped in ``dict()``, or a plain dict)
    and returns exactly:

    ``memory_kind``, ``memory_kind_label``, ``memory_kind_state``
    (one of ``'confirmed' | 'suggested' | 'legacy' | 'untyped'``),
    ``memory_kind_source``, ``memory_kind_confidence``.

    A row with no `memory_kind` of its own, or one whose suggestion falls
    below the display threshold, is shown as the kind nearest its legacy
    ``fact_type`` with state ``'legacy'`` — never as a wrong or invented kind,
    and never by raising. A row with neither a parseable kind nor a mappable
    legacy ``fact_type`` reports ``'untyped'``. Pure read: never classifies,
    never touches storage.
    """
    parsed = parse_kind(_read(obj, "memory_kind"))
    raw_source = _read(obj, "memory_kind_source")
    confidence = _as_float(_read(obj, "memory_kind_confidence"))

    if parsed is not None and is_confirmed(raw_source if isinstance(raw_source, str) else None):
        return {
            "memory_kind": parsed.value,
            "memory_kind_label": LABELS[parsed],
            "memory_kind_state": "confirmed",
            "memory_kind_source": raw_source,
            "memory_kind_confidence": confidence,
        }

    if parsed is not None and confidence is not None and confidence >= display_min_confidence:
        return {
            "memory_kind": parsed.value,
            "memory_kind_label": LABELS[parsed],
            "memory_kind_state": "suggested",
            "memory_kind_source": raw_source if isinstance(raw_source, str) else None,
            "memory_kind_confidence": confidence,
        }

    fact_type = _value_of(_read(obj, "fact_type"))
    legacy_key = str(fact_type).strip().lower() if isinstance(fact_type, str) else None
    legacy_kind = LEGACY_TO_KIND.get(legacy_key) if legacy_key else None
    if legacy_kind is not None:
        return {
            "memory_kind": legacy_kind.value,
            "memory_kind_label": LABELS[legacy_kind],
            "memory_kind_state": "legacy",
            "memory_kind_source": None,
            "memory_kind_confidence": None,
        }

    return {
        "memory_kind": None,
        "memory_kind_label": None,
        "memory_kind_state": "untyped",
        "memory_kind_source": None,
        "memory_kind_confidence": None,
    }
